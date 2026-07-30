// The COMPLETE arandu graph, end-to-end, overlapped + PROFILED:
//   paths -> load -> fingernet ONNX (MODEL phase) -> postproc -> serialize
// PipelineExecutor: model single-stream micro-batched actor(s) overlapped with
// K-way CPU phases; native profiler attributes Wait/Compute/Emit per phase and
// gpu_run/extract inside the model.
//
//   full_graph --onnx M.onnx --images DIR|LIST.txt --out DIR|none [flags]
//
//   --exec pipeline|serial|cpu   default pipeline; `cpu` is the same
//                                PipelineExecutor forced onto the CPU fallback,
//                                which is what the retired StreamExecutor did
//   --provider cuda|tensorrt|cpu, --fp16, --engine-cache DIR   model backend
//   --no-tf32 --conv-algo HEURISTIC|EXHAUSTIVE  the two parity knobs (see onnx_model.hpp)
//   --n N --batch B --threads T --actors A --depth D --warmup W --streams S
//                                resources; --streams 0 forces the model phase onto its
//                                cpu fallback (one image per call), which is a mode, not a bug
//   --threshold F --full         extraction parameters (--full adds the two _mod maps)
//   --device ID --shard I/N      one process per GPU: same command, different I
//   --out none                   skips disk, to isolate compute from I/O
#include <algorithm>
#include <any>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

namespace {

struct Args {
    std::string onnx, images, out = "none", exec = "pipeline", provider = "cuda", engine_cache;
    int n = 0, batch = 8, threads = 16, actors = 2, depth = 4, warmup = 8, device = 0;
    int streams = 2;   // >0 is what activates the MODEL phase; see resolve() in graph.hpp
    int shard = 0, shards = 1;
    float threshold = 0.05f;
    bool fp16 = false, full = false, tf32 = true;
    std::string conv_algo = "HEURISTIC";
};

[[noreturn]] void die(const std::string& msg) {
    fprintf(stderr, "full_graph: %s\n", msg.c_str());
    std::exit(2);
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + k);
            return argv[++i];
        };
        if (k == "--onnx") a.onnx = val();
        else if (k == "--images") a.images = val();
        else if (k == "--out") a.out = val();
        else if (k == "--exec") a.exec = val();
        else if (k == "--provider") a.provider = val();
        else if (k == "--engine-cache") a.engine_cache = val();
        else if (k == "--n") a.n = std::atoi(val().c_str());
        else if (k == "--batch") a.batch = std::atoi(val().c_str());
        else if (k == "--threads") a.threads = std::atoi(val().c_str());
        else if (k == "--actors") a.actors = std::atoi(val().c_str());
        else if (k == "--depth") a.depth = std::atoi(val().c_str());
        else if (k == "--warmup") a.warmup = std::atoi(val().c_str());
        else if (k == "--device") a.device = std::atoi(val().c_str());
        else if (k == "--streams") a.streams = std::atoi(val().c_str());
        else if (k == "--threshold") a.threshold = std::atof(val().c_str());
        else if (k == "--fp16") a.fp16 = true;
        else if (k == "--no-tf32") a.tf32 = false;
        else if (k == "--conv-algo") a.conv_algo = val();
        else if (k == "--full") a.full = true;
        else if (k == "--shard") {
            std::string s = val();
            auto slash = s.find('/');
            if (slash == std::string::npos) die("--shard wants I/N");
            a.shard = std::atoi(s.substr(0, slash).c_str());
            a.shards = std::max(1, std::atoi(s.substr(slash + 1).c_str()));
        }
        else die("unknown flag " + k);
    }
    if (a.onnx.empty() || a.images.empty()) die("need --onnx and --images");
    return a;
}

// A directory (recursive, .png) or a .txt of one path per line. The id is the path
// relative to the input root, so the output tree mirrors the input one -- and for a
// list, where there is no root, the filename stem (which is what api.py writes when
// it cannot form a relative path).
std::vector<PathItem> collect(const std::string& images) {
    std::vector<PathItem> items;
    if (images.size() > 4 && images.compare(images.size() - 4, 4, ".txt") == 0) {
        std::ifstream f(images);
        if (!f) die("cannot read list " + images);
        for (std::string line; std::getline(f, line);) {
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (!line.empty()) items.push_back({line, fs::path(line).stem().string()});
        }
    } else {
        std::vector<std::string> files;
        for (auto& e : fs::recursive_directory_iterator(images))
            if (e.is_regular_file() && e.path().extension() == ".png") files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
        // lexically_relative, not relative: the latter runs both paths through
        // weakly_canonical, which resolves symlinks -- so a symlinked input directory
        // would yield ids full of "../" and scatter the output outside --out.
        for (auto& f : files)
            items.push_back({f, fs::path(f).lexically_relative(images).replace_extension("").string()});
    }
    return items;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    std::vector<PathItem> items = collect(a.images);
    if (a.n > 0 && (int)items.size() > a.n) items.resize(a.n);
    // Shard AFTER truncating, so --n means the same set of images regardless of how
    // many processes split it. Strided, so every shard sees the same size mix.
    if (a.shards > 1) {
        std::vector<PathItem> mine;
        for (size_t i = a.shard; i < items.size(); i += a.shards) mine.push_back(items[i]);
        items.swap(mine);
    }
    if (items.empty()) die("no images");
    std::vector<std::any> input(items.begin(), items.end());

    printf("graph: %zu imgs (shard %d/%d), exec=%s, provider=%s%s, T=%d, B=%d, actors=%d, "
           "depth=%d, streams=%d, warmup=%d, thr=%.3f, tf32=%d, algo=%s%s\n",
           items.size(), a.shard, a.shards, a.exec.c_str(), a.provider.c_str(),
           a.fp16 ? "+fp16" : "", a.threads, a.batch, a.actors, a.depth, a.streams, a.warmup,
           a.threshold, (int)a.tf32, a.conv_algo.c_str(), a.full ? ", full" : "");

    // The padded shape of the first image: the model needs it before the session
    // exists, for the TensorRT profile and the transport declaration.
    int nh = 0, nw = 0;
    {
        std::vector<InputImage> probe(1);
        arandu::RunCtx pc;
        LoadNode{}.run(std::span<const PathItem>(&items[0], 1), probe, pc);
        nh = probe[0].H; nw = probe[0].W;
    }
    printf("padded shape: %dx%d\n", nh, nw);

    auto tl = clk::now();
    FnetOnnxConfig mc;
    mc.path = a.onnx; mc.provider = a.provider; mc.max_batch = a.batch;
    mc.device_id = a.device; mc.fp16 = a.fp16; mc.engine_cache = a.engine_cache;
    mc.threshold = a.threshold; mc.tf32 = a.tf32; mc.conv_algo = a.conv_algo;
    mc.nominal_h = nh; mc.nominal_w = nw;
    // The intra-op pool is per session, so it is sized by how many callers the session
    // will have -- one micro-batched actor per stream, or every cpu worker when the
    // phase falls back (see the gpu_streams note below).
    mc.intra_threads = a.streams > 0 ? 2 : 1;
    auto model = std::make_shared<FingernetOnnx>(mc);
    double t_sess = secs(tl, clk::now());

    auto post = std::make_shared<PostprocNode>(); post->full = a.full;
    arandu::GraphBuilder gb;
    gb.add<PathItem, InputImage>("load", std::make_shared<LoadNode>(), {});
    gb.add_model<InputImage, Bundle>("onnx", model, model, {"load"});
    gb.add<Bundle, Bundle>("postproc", post, {"onnx"});
    gb.add<Bundle, Written>("serialize", std::make_shared<SerializeNode>(a.out, a.full), {"postproc"});
    arandu::Graph g = gb.build();

    // Warmup on the first image, repeated: pays the CUDA/cuDNN/arena cost (and, for
    // TensorRT, the engine build) before the clock starts.
    if (a.warmup > 0) {
        std::vector<PathItem> wp(std::min<size_t>(a.warmup, items.size()), items[0]);
        std::vector<InputImage> wi(wp.size());
        arandu::RunCtx wc; LoadNode{}.run(wp, wi, wc);
        std::vector<Bundle> wb(wi.size());
        model->infer(wi, wb, wc);
    }
    double t_warm = secs(tl, clk::now()) - t_sess;

    arandu::Profiler prof;
    arandu::ExecPolicy pol;
    pol.determinism = arandu::Determinism::Tolerant;
    pol.device = arandu::Device::Cuda;
    pol.cpu_threads = a.threads;
    pol.max_batch = a.batch;
    pol.channel_depth = a.depth;
    // gpu_streams is one of the four conditions arandu::resolve() requires before it
    // picks a phase's IModel over its ICpuScript, and it defaults to 0 -- so a driver
    // that forgets it declares a model phase and silently gets the fallback: not one
    // batched actor, but cpu_threads workers each calling the session with a batch of
    // ONE, with --batch and --actors inert. Nothing complains, because the ONNX
    // session is CUDA either way and the GPU stays busy; it just runs 7% slower
    // (148.4 vs 158.6 img/s, BN48k 1k). Hence the default, and hence the line printed
    // below: which impl each phase resolved to is not something to infer from a
    // throughput number.
    pol.gpu_streams = a.streams;
    pol.model_actors = a.actors;
    pol.profiler = (a.exec == "pipeline") ? &prof : nullptr;

    printf("phases:");
    for (const auto& p : g.phases) {
        arandu::PhaseImpl impl = arandu::resolve(p, pol);
        const char* k = impl == arandu::PhaseImpl::Model ? "MODEL"
                      : impl == arandu::PhaseImpl::Join  ? "join" : "cpu";
        printf("  %s=%s", p.name.c_str(), k);
    }
    printf("\n");

    auto t0 = clk::now();
    std::vector<std::any> res;
    if (a.exec == "pipeline") res = arandu::PipelineExecutor{pol}.run(g, input);
    else if (a.exec == "cpu" || a.exec == "stream") {
        pol.device = arandu::Device::Cpu;
        res = arandu::PipelineExecutor{pol}.run(g, input);
    } else res = arandu::SerialExecutor{}.run(g, input);
    double t_run = secs(t0, clk::now());

    long tot = 0;
    for (auto& r : res) tot += std::any_cast<const Written&>(r).n_minutiae;
    printf("session-create: %.2fs  warmup: %.2fs\n", t_sess, t_warm);
    printf("done: %zu imgs, %ld minutiae, run %.2fs (%.1f img/s)  [+session+warmup -> %.1f img/s]\n",
           res.size(), tot, t_run, res.size() / t_run, res.size() / (t_run + t_sess + t_warm));
    if (pol.profiler) {
        printf("\n%s\n", prof.markdown().c_str());
        if (a.out != "none") {
            std::ofstream(fs::path(a.out) / "trace.json") << prof.chrome_trace();
            printf("chrome trace -> %s/trace.json\n", a.out.c_str());
        }
    }
    return 0;
}
