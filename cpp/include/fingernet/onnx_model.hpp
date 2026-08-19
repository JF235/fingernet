// FingerNet ONNX as an arandu source node: InputImage -> FnetRaw.
// The model has 1 input and 7 outputs (3 float maps + 4 argmax index planes), so it
// can't use arandu's generic single-out OnnxModel<In,Out>; this is the
// fingernet-specific model adapter (fingernet owns its model wiring, arandu builds
// the graph). Batches the input span in chunks of max_batch. Gated on
// FINGERNET_WITH_ONNX.
#pragma once
#ifdef FINGERNET_WITH_ONNX
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
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
    //
    // KNOWN LIMIT, measured: with tf32=0, cuDNN's heuristic asks for a 33 GB workspace
    // on one Conv of a batch-8 512x512 forward. A single actor survives it; two
    // concurrent Run calls on the same session do not, and the arena aborts the
    // process. So the oracle regime needs model_actors=1 (full_graph: --actors 1).
    // cudnn_conv_use_max_workspace=0 does NOT cap it -- tried, same request.
    bool tf32 = true;
    // cuDNN algorithm choice. ORT's own default is EXHAUSTIVE (autotune), which is
    // torch's cudnn.benchmark=True -- and api.py deliberately leaves that OFF because
    // autotune picks per-run algorithms whose reduction order differs. HEURISTIC is
    // the matching setting. It does NOT buy run-to-run reproducibility: measured, two
    // runs of the same binary over 200 BN48k images agree on 178/200 .min files with
    // TF32 on, at any batch size. Reproducibility here needs use_tf32=0.
    std::string conv_algo = "HEURISTIC";  // HEURISTIC | EXHAUSTIVE | DEFAULT
    // ORT's intra-op pool, which runs the nodes the CUDA EP leaves on the host: 15 of
    // the 423, plus the 4 Memcpy that stitch the two sides. Worth 3.5% on CUDA (6.57 ->
    // 6.34 ms/img) and saturating at 2; TensorRT is indifferent, having compiled the
    // whole graph into 3 nodes. It is per SESSION, so the right value depends on how
    // many callers the session has: 2 under the micro-batched actor, 1 when the phase
    // runs on its cpu fallback and every worker thread calls in.
    int intra_threads = 2;
    // How many threads OWN this session -- see the runner below. It is the number of
    // forwards that can be in flight on it at once, so it wants to be model_actors:
    // measured on 512 SD4 images, 76.5 / 85.2 / 82.0 img/s at 1 / 2 / 3 actors.
    int ort_threads = 2;
    // Nominal (padded) input shape, with two jobs that both want the shape this run
    // will actually see:
    //   * the transport declaration (see transport()) -- arandu::TransportSpec is a
    //     per-item promise the profiler reports and can assert, not a buffer size;
    //   * the TensorRT optimization profile. The export has a dynamic batch axis, and
    //     TensorRT builds one engine per shape it MEETS: a run whose micro-batches
    //     happen to be 8,8,5,8,3,... builds an engine for each, mid-run, ~60 s apiece.
    //     That is not TensorRT being slow, it is TensorRT being asked a new question --
    //     it took this pipeline to 9.8 img/s against CUDA's 149. Declaring the profile
    //     as batch 1..max_batch at this shape builds ONE engine that covers every
    //     micro-batch the executor can produce, which is also why the constructor
    //     refuses a graph whose input is not kInputName: the profile is keyed by that
    //     name, and a silent mismatch is a silent return to 9.8 img/s.
    // Defaults to the SD258 padded shape; a driver that knows better should say so.
    int nominal_h = 768;
    int nominal_w = 800;
};

//: The input convert_to_onnx.py names. Not a per-run choice -- a term of the export
//: contract, needed before a session exists to build the TensorRT profile string.
inline constexpr const char* kInputName = "input_image";

// Both ICpuScript (fallback / SerialExecutor) and IModel (PipelineExecutor's
// micro-batched GPU phase via GraphBuilder::add_model).
class FingernetOnnx : public arandu::ICpuScript<InputImage, FnetRaw>,
                      public arandu::IModel<InputImage, FnetRaw> {
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
            std::unordered_map<std::string, std::string> m{
                {"device_id", std::to_string(cfg_.device_id)},
                {"cudnn_conv_algo_search", cfg_.conv_algo},
                {"use_tf32", cfg_.tf32 ? "1" : "0"}};
            o.Update(m);
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
                    return std::string(kInputName) + ":" + std::to_string(b) + "x1x" +
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
        if (in_name_ != kInputName)
            throw std::runtime_error("fingernet onnx: input is '" + in_name_ + "', expected '" +
                                     kInputName + "'; the TensorRT profile would be ignored");
        size_t no = session_->GetOutputCount();
        for (size_t i = 0; i < no; ++i) {
            std::string n = session_->GetOutputNameAllocated(i, a).get();
            out_names_.push_back(n);
            out_idx_[n] = i;
        }
        // PAGE-LOCKED STAGING, WITHOUT A CUDA DEPENDENCY. ORT registers a "CudaPinned"
        // allocator whenever the CUDA EP is loaded, so both directions can go through
        // pinned host memory with no cuda_runtime.h, no cudaHostAlloc and nothing new to
        // link -- which is why this lives here rather than as a copy of mntstitch's
        // PinnedPool. Asking for it is also the test that it exists: on the cpu provider
        // it throws, and then the pageable path stays and transport() SAYS SO. A
        // declaration that claimed pinned while copying pageable is the exact failure the
        // two-flag TransportSpec was introduced to catch.
        try {
            pinned_mem_ = Ort::MemoryInfo("CudaPinned", OrtDeviceAllocator, cfg_.device_id,
                                          OrtMemTypeCPUOutput);
            pinned_alloc_ = std::make_unique<Ort::Allocator>(*session_, pinned_mem_);
            pinned_ok_ = true;
        } catch (const Ort::Exception&) {
            pinned_ok_ = false;                  // cpu provider, or no pinned allocator
        }
        // ── THE SESSION OWNS ITS CALLERS ────────────────────────────────────
        //
        // An ORT CUDA session cannot be Run from a thread that ARRIVES AFTER another one
        // already ran it. Reduced to twenty lines: forward on thread A, join A, forward on
        // thread B -- and B aborts the process with `CUDA failure 400: invalid resource
        // handle` out of cuda_stream_handle.cc's cudaEventRecord. Keeping a third thread
        // alive throughout does not help; it is the ARRIVING thread that fails, not the
        // leaving one. (Concurrent threads are fine: two actors sharing this session is
        // the normal, working case.)
        //
        // That constraint is what stands between this pipeline and a warm session, and a
        // warm session is worth a great deal: the same 24 images take 2.50 s cold and
        // 0.15 s warm -- 9.6 -> 160 img/s -- because cuDNN's algorithm choice and its
        // workspace are per SESSION, paid on the first forwards and thrown away with the
        // process. An executor that spawns fresh workers per run, which is what an executor
        // does, can never collect that.
        //
        // So the session brings its own threads, created here and joined in the destructor,
        // and every Run for its whole life happens on one of them: infer() posts a
        // micro-batch and waits. arandu's actors keep their count and their shape; they
        // simply stop being the threads that talk to CUDA. Measured with two of them, four
        // consecutive rounds of 24 images: 2.50, 0.30, 0.15, 0.15 s.
        //
        // The cpu provider has no such constraint and no reason to pay a hop, so it gets no
        // runner and calls straight through (see forward_on_owner).
        if (cfg_.provider != "cpu") {
            const int n = std::max(1, cfg_.ort_threads);
            for (int i = 0; i < n; ++i) ort_threads_.emplace_back([this] { serve(); });
        }
    }

    ~FingernetOnnx() {
        { std::lock_guard<std::mutex> lk(q_mx_); stopping_ = true; }
        q_cv_.notify_all();
        for (auto& t : ort_threads_) t.join();   // before ~Ort::Session, which follows
    }

    void run(std::span<const InputImage> in, std::span<FnetRaw> out, arandu::RunCtx& ctx) const override {
        forward_on_owner(in, out, ctx);
    }
    // IModel (used by PipelineExecutor's micro-batched MODEL phase)
    void infer(std::span<const InputImage> in, std::span<FnetRaw> out, arandu::RunCtx& ctx) const override {
        forward_on_owner(in, out, ctx);
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
    // Both directions go through PAGE-LOCKED host memory when the CUDA EP gave us its
    // pinned allocator (see the constructor): the input is staged into a pinned block and
    // the seven outputs are bound to the same allocator through IoBinding, so ORT's D2H is
    // a pinned copy too. `pinned_ok_` is reported rather than assumed, because the cpu
    // provider has no such allocator and the flags are read by
    // ExecPolicy::assert_transport_pinned -- a declaration that claimed pinned while
    // copying pageable is worse than the pageable copy.
    //
    // Measured on this model, batch 8 at 512x512, 20 forwards after 3 warmups: 54.32 ms
    // pageable -> 52.31 ms pinned, +3.7%. Small, and small for a good reason -- the trim
    // that dropped the 221 MB orientation map took this from 22 bytes per input pixel to
    // 4.375, so there is not much transfer left to make cheap.
    arandu::TransportSpec transport() const override {
        const std::size_t px = (std::size_t)cfg_.nominal_h * cfg_.nominal_w;
        return {/*d2h*/ px * 4 + px / 64 * 4 * 6, /*h2d*/ px * sizeof(float),
                /*d2h_pinned*/ pinned_ok_, /*h2d_pinned*/ pinned_ok_};
    }

private:
    // ── the runner ──────────────────────────────────────────────────────────

    void serve() const {
        for (;;) {
            std::packaged_task<void()> t;
            {
                std::unique_lock<std::mutex> lk(q_mx_);
                q_cv_.wait(lk, [this] { return stopping_ || !q_.empty(); });
                if (q_.empty()) return;              // stopping, and nothing left to serve
                t = std::move(q_.front());
                q_.pop();
            }
            t();
        }
    }

    /// Run the forward on a thread that OWNS the session, and wait for it. The caller
    /// blocks, so the spans and the RunCtx (profiling sink included) are touched by one
    /// thread at a time -- and the timings still land on the actor that asked, which is
    /// where a reader looks for them.
    void forward_on_owner(std::span<const InputImage> in, std::span<FnetRaw> out,
                          arandu::RunCtx& ctx) const {
        if (ort_threads_.empty()) { forward(in, out, ctx); return; }   // cpu provider
        std::packaged_task<void()> task([this, in, out, &ctx] { forward(in, out, ctx); });
        std::future<void> fut = task.get_future();
        {
            std::lock_guard<std::mutex> lk(q_mx_);
            q_.push(std::move(task));
        }
        q_cv_.notify_one();
        fut.get();                                   // rethrows what the runner threw
    }

    void forward(std::span<const InputImage> in, std::span<FnetRaw> out, arandu::RunCtx& ctx) const {
        const int N = static_cast<int>(in.size());
        for (int b0 = 0; b0 < N; b0 += cfg_.max_batch)
            forward_chunk(in, out, b0, std::min(b0 + cfg_.max_batch, N), ctx);
    }

    void forward_chunk(std::span<const InputImage> in, std::span<FnetRaw> out, int b0, int b1,
                       arandu::RunCtx& ctx) const {
        int B = b1 - b0;
        int H = in[b0].H, W = in[b0].W, HW = H * W;
        const size_t need = static_cast<size_t>(B) * HW * sizeof(float);
        // One staging buffer, from the pool when it is pinned and from the stack when it
        // is not. Whichever it is, the copy-in below and the Run are the same shape.
        Ort::MemoryAllocation blk(nullptr, nullptr, 0);
        std::vector<float> pageable;
        float* stage = nullptr;
        if (pinned_ok_) { blk = borrow(need); stage = static_cast<float*>(blk.get()); }
        else { pageable.resize(static_cast<size_t>(B) * HW); stage = pageable.data(); }
        for (int k = 0; k < B; ++k)
            std::memcpy(stage + static_cast<size_t>(k) * HW, in[b0 + k].data.data(), HW * sizeof(float));
        std::array<int64_t, 4> shp{B, 1, H, W};
        std::vector<Ort::Value> r;
        std::optional<Ort::IoBinding> bind;   // scoped to the call; empty when pageable
        {
            ARANDU_PROF(ctx, "gpu_run");   // ORT Run blocks until outputs ready => GPU+launch
            if (pinned_ok_) {
                Ort::Value inp = Ort::Value::CreateTensor<float>(
                    pinned_mem_, stage, static_cast<size_t>(B) * HW, shp.data(), 4);
                bind.emplace(*session_);
                bind->BindInput(in_name_.c_str(), inp);
                // Bound to the ALLOCATOR, not to a buffer: the coarse heads' shape depends
                // on the input, and letting ORT allocate each output page-locked keeps the
                // seven shapes the export's business instead of a table to maintain here.
                for (const auto& n : out_names_) bind->BindOutput(n.c_str(), pinned_mem_);
                session_->Run(Ort::RunOptions{nullptr}, *bind);
                r = bind->GetOutputValues();   // same order as out_names_ => out_idx_ holds
            } else {
                auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value inp = Ort::Value::CreateTensor<float>(
                    mem, stage, static_cast<size_t>(B) * HW, shp.data(), 4);
                const char* inn[]{in_name_.c_str()};
                std::vector<const char*> onn;
                for (auto& n : out_names_) onn.push_back(n.c_str());
                r = session_->Run(Ort::RunOptions{nullptr}, inn, &inp, 1, onn.data(), onn.size());
            }
        }
        ARANDU_PROF(ctx, "extract");       // host-side copy of the 7 outputs, per image

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
            out[b0 + k] = FnetRaw{std::move(ri)};
        }
        // Only on the way out clean: a throw above frees the block through blk's destructor
        // instead of pooling it, which loses a buffer and never a correctness property.
        if (pinned_ok_) give_back(std::move(blk));
    }

    // ── the pinned staging pool ─────────────────────────────────────────────
    //
    // Page-LOCKING is what costs, not the bytes: through ORT's arena an 8 MiB
    // alloc+free measures 0.105 ms, which is cheap but not free at ~16 chunks/s per
    // actor, and a fresh cudaHostAlloc would be far worse. So blocks are reused. The
    // free-list is bounded (a run whose padded shape changes must not hoard one block
    // per shape) and a block too small for the request is simply not taken -- the
    // borrow allocates, and the oversized one keeps serving the batches it fits.
    static constexpr size_t kPoolBlocks = 8;   // >= model_actors of any real policy

    Ort::MemoryAllocation borrow(size_t bytes) const {
        {
            std::lock_guard<std::mutex> lk(pool_mx_);
            for (size_t i = 0; i < pool_.size(); ++i)
                if (pool_[i].size() >= bytes) {
                    Ort::MemoryAllocation blk = std::move(pool_[i]);
                    pool_.erase(pool_.begin() + static_cast<std::ptrdiff_t>(i));
                    return blk;
                }
        }
        return pinned_alloc_->GetAllocation(bytes);
    }
    void give_back(Ort::MemoryAllocation blk) const {
        std::lock_guard<std::mutex> lk(pool_mx_);
        if (pool_.size() < kPoolBlocks) pool_.push_back(std::move(blk));
        // else: the destructor frees it here, which is the bound doing its job.
    }

    FnetOnnxConfig cfg_;
    Ort::Env env_;
    mutable std::unique_ptr<Ort::Session> session_;
    std::string in_name_;
    std::vector<std::string> out_names_;
    std::unordered_map<std::string, size_t> out_idx_;
    Ort::MemoryInfo pinned_mem_{nullptr};
    std::unique_ptr<Ort::Allocator> pinned_alloc_;
    bool pinned_ok_ = false;
    mutable std::mutex pool_mx_;
    mutable std::vector<Ort::MemoryAllocation> pool_;
    // The session's own threads and the queue they serve (see the constructor).
    mutable std::mutex q_mx_;
    mutable std::condition_variable q_cv_;
    mutable std::queue<std::packaged_task<void()>> q_;
    mutable bool stopping_ = false;
    std::vector<std::thread> ort_threads_;
};

}  // namespace fnaru
#endif  // FINGERNET_WITH_ONNX
