// FingerNet trimmed ONNX as an arandu source node: InputImage -> Bundle(raw).
// The model has 1 input and 7 outputs, so it can't use arandu's generic
// single-out OnnxModel<In,Out>; this is the fingernet-specific model adapter
// (fingernet owns its model wiring, arandu builds the graph). Batches the input
// span in chunks of max_batch. Gated on FINGERNET_WITH_ONNX.
#pragma once
#ifdef FINGERNET_WITH_ONNX
#include <algorithm>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "arandu/kernel.hpp"
#include "arandu/profiling.hpp"
#include "arandu_nodes.hpp"

namespace fnaru {

struct FnetOnnxConfig {
    std::string path;
    std::string provider = "cuda";      // cpu | cuda | tensorrt
    int device_id = 0;
    int max_batch = 8;
    bool fp16 = false;
    std::string engine_cache;
    // Nominal input shape, for the transport declaration ONLY (see transport()). The real
    // shape is per-image; arandu::TransportSpec is a per-item promise the profiler reports
    // and can assert, not a buffer size. Defaults to the SD258 padded shape.
    int nominal_h = 768;
    int nominal_w = 800;
};

// Both ICpuScript (fallback / SerialExecutor) and IModel (PipelineExecutor's
// micro-batched GPU phase via GraphBuilder::add_model).
class FingernetOnnx : public arandu::ICpuScript<InputImage, Bundle>,
                      public arandu::IModel<InputImage, Bundle> {
public:
    explicit FingernetOnnx(FnetOnnxConfig cfg)
        : cfg_(std::move(cfg)), env_(ORT_LOGGING_LEVEL_WARNING, "fingernet") {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(1);
        so.SetInterOpNumThreads(1);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (cfg_.provider == "cuda") {
            OrtCUDAProviderOptions o{}; o.device_id = cfg_.device_id;
            so.AppendExecutionProvider_CUDA(o);
        } else if (cfg_.provider == "tensorrt" || cfg_.provider == "trt") {
            OrtTensorRTProviderOptions o{}; o.device_id = cfg_.device_id;
            o.trt_fp16_enable = cfg_.fp16 ? 1 : 0;
            if (!cfg_.engine_cache.empty()) { o.trt_engine_cache_enable = 1; o.trt_engine_cache_path = cfg_.engine_cache.c_str(); }
            so.AppendExecutionProvider_TensorRT(o);
            OrtCUDAProviderOptions c{}; c.device_id = cfg_.device_id;
            so.AppendExecutionProvider_CUDA(c);
        }
        session_ = std::make_unique<Ort::Session>(env_, cfg_.path.c_str(), so);
        Ort::AllocatorWithDefaultOptions a;
        in_name_ = session_->GetInputNameAllocated(0, a).get();
        size_t no = session_->GetOutputCount();
        for (size_t i = 0; i < no; ++i) {
            std::string n = session_->GetOutputNameAllocated(i, a).get();
            out_names_.push_back(n);
            out_idx_[n] = i;
        }
    }

    void run(std::span<const InputImage> in, std::span<Bundle> out, arandu::RunCtx& ctx) const override {
        forward(in, out, ctx);
    }
    // IModel (used by PipelineExecutor's micro-batched MODEL phase)
    void infer(std::span<const InputImage> in, std::span<Bundle> out, arandu::RunCtx& ctx) const override {
        forward(in, out, ctx);
    }
    int max_batch() const override { return cfg_.max_batch; }
    bool bit_exact_batch_invariant() const override { return cfg_.provider == "cpu"; }

    // What crosses the device boundary per image. arandu::IModel::transport() has no
    // default on purpose: this model is the reason. Its full ONNX export shipped an
    // orientation map at INPUT resolution (~221 MB/image) whose D2H cost 127 ms against
    // 15 ms of compute -- 8x the forward, spent inside the same call, so it read as slow
    // inference on an idle GPU. Summing the output shapes once, here, is what makes that
    // visible instead of mysterious.
    //
    // The coarse heads live on a stride-8 grid (hw = HW/64); enhanced_real is full-res:
    //   segmentation hw + orientation 90*hw + minutiae_orientation 180*hw
    //   + minutiae_{x,y}_offset 8*hw each + minutiae_score hw   = 288*hw
    //   + enhanced_real HW
    // = 288*HW/64 + HW = 5.5*HW floats = 22 bytes per input pixel.
    //
    // Both directions are PAGEABLE and declared as such: the input is staged through a
    // plain std::vector<float> in forward_chunk, and the outputs come back through ORT's
    // default host allocator. Neither is pinned, and saying so is the point -- a pageable
    // bulk copy runs at ~6 GB/s where pinned reaches ~28.
    arandu::TransportSpec transport() const override {
        const std::size_t px = (std::size_t)cfg_.nominal_h * cfg_.nominal_w;
        return {/*d2h*/ px * 22, /*h2d*/ px * sizeof(float),
                /*d2h_pinned*/ false, /*h2d_pinned*/ false};
    }

private:
    void forward(std::span<const InputImage> in, std::span<Bundle> out, arandu::RunCtx& ctx) const {
        const int N = static_cast<int>(in.size());
        for (int b0 = 0; b0 < N; b0 += cfg_.max_batch)
            forward_chunk(in, out, b0, std::min(b0 + cfg_.max_batch, N), ctx);
    }

    void forward_chunk(std::span<const InputImage> in, std::span<Bundle> out, int b0, int b1,
                       arandu::RunCtx& ctx) const {
        int B = b1 - b0;
        int H = in[b0].H, W = in[b0].W, HW = H * W;
        std::vector<float> buf(static_cast<size_t>(B) * HW);
        for (int k = 0; k < B; ++k)
            std::memcpy(buf.data() + static_cast<size_t>(k) * HW, in[b0 + k].data.data(), HW * sizeof(float));
        std::array<int64_t, 4> shp{B, 1, H, W};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inp = Ort::Value::CreateTensor<float>(mem, buf.data(), buf.size(), shp.data(), 4);
        const char* inn[]{in_name_.c_str()};
        std::vector<const char*> onn;
        for (auto& n : out_names_) onn.push_back(n.c_str());
        std::vector<Ort::Value> r;
        {
            ARANDU_PROF(ctx, "gpu_run");   // ORT Run blocks until outputs ready => GPU+launch
            r = session_->Run(Ort::RunOptions{nullptr}, inn, &inp, 1, onn.data(), onn.size());
        }
        ARANDU_PROF(ctx, "extract");       // host-side copy of the 7 outputs into Bundles

        auto get = [&](const char* name) -> const float* {
            return r[out_idx_.at(name)].GetTensorData<float>();
        };
        int h = in[b0].h, w = in[b0].w, hw = h * w;
        const float* p_seg = get("segmentation");
        const float* p_ori = get("orientation");
        const float* p_enh = get("enhanced_real");
        const float* p_mo = get("minutiae_orientation");
        const float* p_mx = get("minutiae_x_offset");
        const float* p_my = get("minutiae_y_offset");
        const float* p_ms = get("minutiae_score");
        for (int k = 0; k < B; ++k) {
            auto ri = std::make_shared<RawImage>();
            ri->h = h; ri->w = w; ri->H = H; ri->W = W; ri->threshold = threshold_;
            ri->id = in[b0 + k].id; ri->orig_h = in[b0 + k].orig_h; ri->orig_w = in[b0 + k].orig_w;
            auto cp = [&](const float* src, int per) {
                return std::vector<float>(src + static_cast<size_t>(k) * per, src + static_cast<size_t>(k + 1) * per);
            };
            ri->segmentation = cp(p_seg, hw);
            ri->orientation = cp(p_ori, 90 * hw);
            ri->enhanced_real = cp(p_enh, HW);
            ri->minutiae_orientation = cp(p_mo, 180 * hw);
            ri->minutiae_x_offset = cp(p_mx, 8 * hw);
            ri->minutiae_y_offset = cp(p_my, 8 * hw);
            ri->minutiae_score = cp(p_ms, hw);
            Bundle bd; bd.raw = ri;
            out[b0 + k] = std::move(bd);
        }
    }

    FnetOnnxConfig cfg_;
    float threshold_ = 0.05f;
    Ort::Env env_;
    mutable std::unique_ptr<Ort::Session> session_;
    std::string in_name_;
    std::vector<std::string> out_names_;
    std::unordered_map<std::string, size_t> out_idx_;
};

}  // namespace fnaru
#endif  // FINGERNET_WITH_ONNX
