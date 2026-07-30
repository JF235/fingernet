// FingerNet ONNX as an arandu source node: InputImage -> Bundle(raw).
// The model has 1 input and 7 outputs (3 float maps + 4 argmax index planes), so it
// can't use arandu's generic single-out OnnxModel<In,Out>; this is the
// fingernet-specific model adapter (fingernet owns its model wiring, arandu builds
// the graph). Batches the input span in chunks of max_batch. Gated on
// FINGERNET_WITH_ONNX.
#pragma once
#ifdef FINGERNET_WITH_ONNX
#include <algorithm>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <stdexcept>
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
    bool fp16 = false;                  // TensorRT only
    std::string engine_cache;
    float threshold = 0.05f;            // minutia score gate; api.py's default
    // TF32 for fp32 convolutions. ON is the throughput choice AND what api.py runs
    // (fp32_precision = "tf32"), so it is the default here too -- but it is the one
    // knob that decides whether this pipeline can be compared byte-for-byte with
    // anything: TF32 has ~10 mantissa bits, and the divergence it introduces is
    // larger than every other difference between the two tiers put together. Measured
    // against a torch-CPU fp32 reference on 8 BN48k images: with TF32 the argmax
    // planes agree 99.3-99.9% (torch-cuda and ORT-cuda each landing somewhere in that
    // band, differently); with use_tf32=0 + HEURISTIC they agree 100.000% and the
    // float maps to 7.6e-6. So: leave it on to extract, turn it off to compare.
    bool tf32 = true;
    // cuDNN algorithm choice. ORT's own default is EXHAUSTIVE (autotune), which is
    // torch's cudnn.benchmark=True -- and api.py deliberately leaves that OFF because
    // autotune picks per-run algorithms whose reduction order differs. HEURISTIC is
    // the matching setting, and it also makes this tier reproducible run to run.
    std::string conv_algo = "HEURISTIC";  // HEURISTIC | EXHAUSTIVE | DEFAULT
    // ORT's intra-op pool, which runs the nodes the CUDA EP leaves on the host: 15 of
    // the 423, plus the 4 Memcpy that stitch the two sides. Worth 3.5% on CUDA (6.57 ->
    // 6.34 ms/img) and saturating at 2; TensorRT is indifferent, having compiled the
    // whole graph into 3 nodes. It is per SESSION, so the right value depends on how
    // many callers the session has: 2 under the micro-batched actor, 1 when the phase
    // runs on its cpu fallback and every worker thread calls in.
    int intra_threads = 2;
    // The graph's input name, needed to declare the TensorRT profile BEFORE the session
    // exists (which is the only place the session could tell us). convert_to_onnx.py
    // sets it; the constructor asserts the session agrees, so a rename fails loudly
    // instead of silently dropping the profile and going back to 9.8 img/s.
    std::string input_name = "input_image";
    // Nominal (padded) input shape. Two jobs, both of which want the shape this run
    // will actually see:
    //   * the transport declaration (see transport()) -- arandu::TransportSpec is a
    //     per-item promise the profiler reports and can assert, not a buffer size;
    //   * the TensorRT optimization profile. The export has a dynamic batch axis, and
    //     TensorRT builds one engine per shape it MEETS: a run whose micro-batches
    //     happen to be 8,8,5,8,3,... builds an engine for each, mid-run, ~60 s apiece.
    //     That is not TensorRT being slow, it is TensorRT being asked a new question --
    //     it took this pipeline to 9.8 img/s against CUDA's 149. Declaring the profile
    //     as batch 1..max_batch at this shape builds ONE engine that covers every
    //     micro-batch the executor can produce.
    // Defaults to the SD258 padded shape; a driver that knows better should say so.
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
        so.SetIntraOpNumThreads(cfg_.intra_threads);
        so.SetInterOpNumThreads(1);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // V2 (string-keyed) provider options, not the legacy structs: use_tf32 has no
        // field in OrtCUDAProviderOptions, and it is the setting that decides parity.
        auto append_cuda = [&] {
            Ort::CUDAProviderOptions o;
            o.Update({{"device_id", std::to_string(cfg_.device_id)},
                      {"cudnn_conv_algo_search", cfg_.conv_algo},
                      {"use_tf32", cfg_.tf32 ? "1" : "0"}});
            so.AppendExecutionProvider_CUDA_V2(*o);
        };
        if (cfg_.provider == "cuda") {
            append_cuda();
        } else if (cfg_.provider == "tensorrt" || cfg_.provider == "trt") {
            Ort::TensorRTProviderOptions t;
            std::unordered_map<std::string, std::string> m{
                {"device_id", std::to_string(cfg_.device_id)},
                {"trt_fp16_enable", cfg_.fp16 ? "1" : "0"}};
            if (!cfg_.engine_cache.empty()) {
                m["trt_engine_cache_enable"] = "1";
                m["trt_engine_cache_path"] = cfg_.engine_cache;
            }
            if (cfg_.nominal_h > 0 && cfg_.nominal_w > 0) {
                auto shape = [&](int b) {
                    return cfg_.input_name + ":" + std::to_string(b) + "x1x" +
                           std::to_string(cfg_.nominal_h) + "x" + std::to_string(cfg_.nominal_w);
                };
                m["trt_profile_min_shapes"] = shape(1);
                m["trt_profile_opt_shapes"] = shape(cfg_.max_batch);
                m["trt_profile_max_shapes"] = shape(cfg_.max_batch);
            }
            t.Update(m);
            so.AppendExecutionProvider_TensorRT_V2(*t);
            append_cuda();   // fallback for any node TRT does not claim
        }
        session_ = std::make_unique<Ort::Session>(env_, cfg_.path.c_str(), so);
        Ort::AllocatorWithDefaultOptions a;
        in_name_ = session_->GetInputNameAllocated(0, a).get();
        if (in_name_ != cfg_.input_name)
            throw std::runtime_error("fingernet onnx: input is '" + in_name_ + "' but config says '" +
                                     cfg_.input_name + "'; the TensorRT profile would be ignored");
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
    // default on purpose: this model is the reason. Its first ONNX export shipped an
    // orientation map at INPUT resolution (~221 MB/image) whose D2H cost 127 ms against
    // 15 ms of compute -- 8x the forward, spent inside the same call, so it read as slow
    // inference on an idle GPU. Summing the output shapes once, here, is what makes that
    // visible instead of mysterious.
    //
    // Dropping that map left 288 coarse channels, of which 286 existed only to be
    // argmax'd; that argmax now runs in the graph. The coarse heads live on a stride-8
    // grid (hw = HW/64), enhanced_real is full-res:
    //   segmentation hw + minutiae_score hw + 4 index planes (int32) hw each = 6*hw
    //   + enhanced_real HW
    // = 4*(6*HW/64 + HW) = 4.375 bytes per input pixel, down from 22.
    //
    // Both directions are PAGEABLE and declared as such: the input is staged through a
    // plain std::vector<float> in forward_chunk, and the outputs come back through ORT's
    // default host allocator. Neither is pinned, and saying so is the point -- a pageable
    // bulk copy runs at ~6 GB/s where pinned reaches ~28.
    arandu::TransportSpec transport() const override {
        const std::size_t px = (std::size_t)cfg_.nominal_h * cfg_.nominal_w;
        return {/*d2h*/ px * 4 + px / 64 * 4 * 6, /*h2d*/ px * sizeof(float),
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

        int h = in[b0].h, w = in[b0].w, hw = h * w;
        auto fp = [&](const char* name) { return r[out_idx_.at(name)].GetTensorData<float>(); };
        auto ip = [&](const char* name) { return r[out_idx_.at(name)].GetTensorData<int32_t>(); };
        const float* p_seg = fp("segmentation");
        const float* p_enh = fp("enhanced_real");
        const float* p_ms = fp("minutiae_score");
        const int32_t* p_oi = ip("orientation_index");
        const int32_t* p_mi = ip("minutiae_orientation_index");
        const int32_t* p_xi = ip("minutiae_x_index");
        const int32_t* p_yi = ip("minutiae_y_index");
        for (int k = 0; k < B; ++k) {
            auto ri = std::make_shared<RawImage>();
            ri->h = h; ri->w = w; ri->H = H; ri->W = W; ri->threshold = cfg_.threshold;
            ri->id = in[b0 + k].id; ri->orig_h = in[b0 + k].orig_h; ri->orig_w = in[b0 + k].orig_w;
            auto cp = [&]<class T>(const T* src, int per) {
                return std::vector<T>(src + static_cast<size_t>(k) * per, src + static_cast<size_t>(k + 1) * per);
            };
            ri->segmentation = cp(p_seg, hw);
            ri->enhanced_real = cp(p_enh, HW);
            ri->minutiae_score = cp(p_ms, hw);
            ri->orientation_index = cp(p_oi, hw);
            ri->minutiae_orientation_index = cp(p_mi, hw);
            ri->minutiae_x_index = cp(p_xi, hw);
            ri->minutiae_y_index = cp(p_yi, hw);
            Bundle bd; bd.raw = ri;
            out[b0 + k] = std::move(bd);
        }
    }

    FnetOnnxConfig cfg_;
    Ort::Env env_;
    mutable std::unique_ptr<Ort::Session> session_;
    std::string in_name_;
    std::vector<std::string> out_names_;
    std::unordered_map<std::string, size_t> out_idx_;
};

}  // namespace fnaru
#endif  // FINGERNET_WITH_ONNX
