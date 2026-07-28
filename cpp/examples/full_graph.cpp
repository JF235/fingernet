// The COMPLETE arandu graph, end-to-end, overlapped + PROFILED:
//   paths -> load -> fingernet ONNX (MODEL phase) -> postproc -> serialize
// PipelineExecutor: model single-stream micro-batched actor(s) overlapped with
// K-way CPU phases; native profiler attributes Wait/Compute/Emit per phase and
// gpu_run/extract inside the model. Usage:
//   full_graph <onnx> <imgs> <out> [N] [exec] [T] [provider] [B] [warmup] [model_actors]
//   exec: pipeline (default) | serial | cpu ;  out="none" skips disk
//   (`cpu` is the same PipelineExecutor forced onto the CPU fallback -- it is what the
//    retired StreamExecutor did, and `stream` is still accepted for that.)
#include <algorithm>
#include <any>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "arandu/graph.hpp"
#include "arandu/pipeline_executor.hpp"
#include "arandu/profiling.hpp"
#include "arandu/serial_executor.hpp"
#include "fingernet/arandu_nodes.hpp"
#include "fingernet/io_nodes.hpp"
#include "fingernet/onnx_model.hpp"

namespace fs = std::filesystem;
using namespace fnaru;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

int main(int argc, char** argv) {
    std::string onnx = argv[1], imgs = argv[2], out = argv[3];
    int N = argc > 4 ? std::atoi(argv[4]) : 64;
    std::string exec = argc > 5 ? argv[5] : "pipeline";
    int T = argc > 6 ? std::atoi(argv[6]) : 12;
    std::string provider = argc > 7 ? argv[7] : "cuda";
    int B = argc > 8 ? std::atoi(argv[8]) : 8;
    int warmup = argc > 9 ? std::atoi(argv[9]) : 0;
    int model_actors = argc > 10 ? std::atoi(argv[10]) : 1;

    std::vector<std::string> files;
    for (auto& e : fs::recursive_directory_iterator(imgs))
        if (e.is_regular_file() && e.path().extension() == ".png") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    if ((int)files.size() > N) files.resize(N);
    std::vector<std::any> input;
    for (auto& f : files)
        input.emplace_back(PathItem{f, fs::relative(f, imgs).replace_extension("").string()});
    printf("complete graph: %zu imgs, exec=%s, T=%d, B=%d, model_actors=%d, warmup=%d, provider=%s\n",
           files.size(), exec.c_str(), T, B, model_actors, warmup, provider.c_str());

    auto tl = clk::now();
    FnetOnnxConfig mc; mc.path = onnx; mc.provider = provider; mc.max_batch = B;
    auto model = std::make_shared<FingernetOnnx>(mc);
    double t_sess = secs(tl, clk::now());

    arandu::GraphBuilder gb;
    gb.add<PathItem, InputImage>("load", std::make_shared<LoadNode>(), {});
    gb.add_model<InputImage, Bundle>("onnx", model, model, {"load"});
    gb.add<Bundle, Bundle>("postproc", std::make_shared<PostprocNode>(), {"onnx"});
    gb.add<Bundle, Written>("serialize", std::make_shared<SerializeNode>(out), {"postproc"});
    arandu::Graph g = gb.build();

    // optional warmup: run the model on a small batch to pay CUDA/cuDNN/arena cost
    if (warmup > 0) {
        std::vector<PathItem> wp(files.begin(), files.begin() + std::min<int>(warmup, files.size()));
        for (auto& p : wp) p = wp[0];  // reuse first path
        std::vector<InputImage> wi(wp.size());
        arandu::RunCtx wc; LoadNode{}.run(wp, wi, wc);
        std::vector<Bundle> wb(wi.size());
        model->infer(wi, wb, wc);
    }

    arandu::Profiler prof;
    arandu::ExecPolicy pol;
    pol.determinism = arandu::Determinism::Tolerant;
    pol.device = arandu::Device::Cuda;
    pol.cpu_threads = T;
    pol.max_batch = B;
    pol.channel_depth = 4;
    pol.model_actors = model_actors;
    pol.profiler = (exec == "pipeline") ? &prof : nullptr;

    auto t0 = clk::now();
    std::vector<std::any> res;
    if (exec == "pipeline")     res = arandu::PipelineExecutor{pol}.run(g, input);
    else if (exec == "cpu" || exec == "stream") {
        // The CPU fallback of the same executor: with device=Cpu every phase resolves to
        // its cpu impl, which is exactly what the retired StreamExecutor was.
        pol.device = arandu::Device::Cpu;
        res = arandu::PipelineExecutor{pol}.run(g, input);
    }
    else                        res = arandu::SerialExecutor{}.run(g, input);
    double t_run = secs(t0, clk::now());

    long tot = 0;
    for (auto& a : res) tot += std::any_cast<const Written&>(a).n_minutiae;
    printf("session-create: %.2fs\n", t_sess);
    printf("done: %zu imgs, %ld minutiae, run %.2fs (%.1f img/s)  [+session -> %.1f img/s]\n",
           res.size(), tot, t_run, res.size() / t_run, res.size() / (t_run + t_sess));
    if (pol.profiler) {
        printf("\n%s\n", prof.markdown().c_str());
        if (out != "none") {
            std::ofstream(fs::path(out) / "trace.json") << prof.chrome_trace();
            printf("chrome trace -> %s/trace.json\n", out.c_str());
        }
    }
    return 0;
}
