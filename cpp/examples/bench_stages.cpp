// Stage-separated throughput of the complete pipeline, measured at the CORRECT
// concurrency: model single-stream (batched), postproc + I/O parallel across
// items. The overlapped pipeline is bounded by the slowest stage (min).
// Usage: bench_stages <onnx> <images_dir> <out_dir> [num_imgs] [provider] [threads]
#include <algorithm>
#include <any>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "fingernet/pipeline.hpp"

namespace fs = std::filesystem;
using namespace fnaru;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

int main(int argc, char** argv) {
    std::string onnx = argv[1], imgs = argv[2], out = argv[3];
    int N = argc > 4 ? std::atoi(argv[4]) : 256;
    std::string provider = argc > 5 ? argv[5] : "cuda";
    int T = argc > 6 ? std::atoi(argv[6]) : 8;

    std::vector<std::string> files;
    for (auto& e : fs::recursive_directory_iterator(imgs))
        if (e.is_regular_file() && e.path().extension() == ".png") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    if ((int)files.size() > N) files.resize(N);
    N = (int)files.size();
    printf("stage bench: %d imgs, provider=%s, postproc/IO threads=%d\n", N, provider.c_str(), T);

    // load (once)
    std::vector<PathItem> paths;
    for (auto& f : files) paths.push_back({f, fs::relative(f, imgs).replace_extension("").string()});
    std::vector<InputImage> imgsv(N);
    arandu::RunCtx ctx;
    LoadNode{}.run(paths, imgsv, ctx);

    // ---- Stage 1: MODEL (single stream, batched) --------------------------
    FnetOnnxConfig mc; mc.path = onnx; mc.provider = provider; mc.max_batch = 8;
    FingernetOnnx model(mc);
    std::vector<Bundle> raw(N);
    model.run(imgsv, raw, ctx);                              // warmup (build engine / cudnn autotune)
    auto t0 = clk::now();
    model.run(imgsv, raw, ctx);
    double t_model = secs(t0, clk::now());

    std::vector<std::any> raw_any;
    for (auto& b : raw) raw_any.emplace_back(b);

    // ---- Stage 2: POSTPROC (parallel across items) ------------------------
    arandu::Graph gpost = build_postproc_graph();
    auto pol = arandu::ExecPolicy::elft_submission(); pol.cpu_threads = T;
    arandu::PipelineExecutor se(pol);   // StreamExecutor was retired into this one
    auto post_any = se.run(gpost, raw_any);                  // warmup
    t0 = clk::now();
    post_any = se.run(gpost, raw_any);
    double t_post = secs(t0, clk::now());

    // ---- Stage 3: I/O (parallel across items, PNG+.min to disk) -----------
    arandu::GraphBuilder gb;
    gb.add<Bundle, Written>("serialize", std::make_shared<SerializeNode>(out), {});
    arandu::Graph gio = gb.build();
    t0 = clk::now();
    auto io_any = se.run(gio, post_any);
    double t_io = secs(t0, clk::now());

    auto rate = [&](double t) { return N / t; };
    printf("\n== stage throughput (%d imgs) ==\n", N);
    printf("  model  (ONNX-%s, 1 stream, b8) : %6.2fs  %7.1f img/s\n", provider.c_str(), t_model, rate(t_model));
    printf("  postproc (C++, T=%d)            : %6.2fs  %7.1f img/s\n", T, t_post, rate(t_post));
    printf("  I/O (8 PNG+.min via libpng,T=%d): %6.2fs  %7.1f img/s\n", T, t_io, rate(t_io));
    double slow = std::max({t_model, t_post, t_io});
    printf("  -> pipeline-bound (overlapped) : %7.1f img/s (bottleneck = slowest stage)\n", rate(slow));
    printf("  -> naive sum (no overlap)      : %7.1f img/s\n", N / (t_model + t_post + t_io));
    return 0;
}
