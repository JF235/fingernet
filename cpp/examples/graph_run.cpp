// graph_run: build the fingernet extraction graph from a SPEC FILE and run it.
//
// Same nodes as full_graph and the same executor; the difference is that the WIRING IS
// DATA. full_graph hardcodes load -> onnx -> postproc -> serialize in C++, so a UI that
// wants to offer a different graph has nothing to talk to. This registers the fingernet
// node types with arandu::NodeRegistry and assembles whatever the spec asks for --
// which means the graph a user draws is the graph that runs, and arandu's own
// build() (edge typecheck, cycle detection, single sink) is the validator.
//
// SPEC FORMAT -- deliberately line-oriented, not JSON. The producer is a Python server
// that has JSON for free, and a hand-rolled JSON parser here would be the only parser
// in this repo with no test and nothing else consuming it. `#` comments, blank lines
// ignored, everything TAB-separated:
//
//   @images   <dir|list.txt>        run-level settings, one per line
//   @exec     pipeline|serial
//   @threads  12
//   ...
//   <name>    <type>    <in1,in2>   <k=v;k=v>      one node per line
//
// A run is therefore ONE artifact: the graph, its parameters and the policy in a single
// file that can be kept, diffed and replayed. Usage:
//
//   graph_run <spec-file> [--profile]
//
// Prints a JSON summary on stdout (the server reads it) and, with --profile, the
// profiler's markdown on stderr.
#include <algorithm>
#include <any>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "arandu/registry.hpp"
#include "fingernet/pipeline.hpp"

namespace fs = std::filesystem;
using namespace fnaru;
using clk = std::chrono::steady_clock;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "graph_run: %s\n", msg.c_str());
    std::exit(2);
}

double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, sep);)
        if (!t.empty()) out.push_back(t);
    return out;
}

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t'))
        s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

/// Escape for the JSON summary. Only what a path or an id can contain.
std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

// ── the spec ────────────────────────────────────────────────────────────────

struct Spec {
    std::map<std::string, std::string> run;          // @key -> value
    std::vector<arandu::NodeConfig> nodes;

    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = run.find(k);
        return it == run.end() ? def : it->second;
    }
    int get_int(const std::string& k, int def) const {
        auto it = run.find(k);
        return it == run.end() ? def : std::stoi(it->second);
    }
    bool get_bool(const std::string& k, bool def) const {
        auto it = run.find(k);
        if (it == run.end()) return def;
        return it->second == "1" || it->second == "true" || it->second == "on";
    }
};

Spec read_spec(const std::string& path) {
    std::ifstream f(path);
    if (!f) die("cannot read spec " + path);
    Spec spec;
    int lineno = 0;
    for (std::string raw; std::getline(f, raw);) {
        ++lineno;
        const std::string line = trim(raw);
        if (line.empty() || line[0] == '#') continue;

        if (line[0] == '@') {
            const size_t sep = line.find_first_of(" \t");
            if (sep == std::string::npos) die("line " + std::to_string(lineno) +
                                             ": @setting needs a value");
            spec.run[line.substr(1, sep - 1)] = trim(line.substr(sep + 1));
            continue;
        }

        const auto cols = split(line, '\t');
        if (cols.size() < 2)
            die("line " + std::to_string(lineno) + ": need at least name<TAB>type");
        arandu::NodeConfig c;
        c.name = trim(cols[0]);
        c.type = trim(cols[1]);
        if (cols.size() > 2) c.inputs = split(trim(cols[2]), ',');
        if (cols.size() > 3) {
            for (const auto& kv : split(trim(cols[3]), ';')) {
                const size_t eq = kv.find('=');
                if (eq == std::string::npos)
                    die("line " + std::to_string(lineno) + ": param '" + kv + "' is not k=v");
                c.params[trim(kv.substr(0, eq))] = trim(kv.substr(eq + 1));
            }
        }
        spec.nodes.push_back(std::move(c));
    }
    if (spec.nodes.empty()) die("spec declares no nodes");
    return spec;
}

// ── the node catalog ────────────────────────────────────────────────────────
//
// One registration per fingernet node type. This is the list a UI's palette mirrors,
// and the reason a drawn graph can be built at all. The input/output types are the
// template arguments right here -- which is also why the catalog the UI shows is, for
// now, a second copy in the server: the registry records factories, not signatures.
// When it learns to record them, THIS is the place they come from.

/// The padded shape of the first image. The ONNX node needs it before a session exists
/// (TensorRT profile + transport declaration), and a factory has no other way to reach
/// run-level facts, so it is parked here by main() before build().
struct ProbeShape { int h = 0, w = 0; };
ProbeShape probe_shape;

/// Which GPU this run uses, from the RUN settings (`@gpu`) and not from a node.
///
/// Which GPU is POLICY, like threads and batch: the same graph runs on 0 today and on 3
/// tomorrow, and a node that names a device makes "run this somewhere else" an edit to the
/// pipeline. It lives here rather than in `pol` because the node factory runs inside
/// `build()` -- before a policy exists -- and the ONNX session is created in the model's
/// constructor. Same idiom as `probe_shape` above, for the same reason.
int run_gpu = 0;

bool truthy(const std::string& v) { return v == "1" || v == "true" || v == "on"; }

/// Arity, checked with the node's NAME in the message. GraphBuilder checks it too, but
/// only after the factory has already indexed c.inputs -- so a JOIN wired with one
/// input would go out of bounds before the graph could reject it.
void need_inputs(const arandu::NodeConfig& c, size_t n) {
    if (c.inputs.size() != n)
        throw std::runtime_error("node '" + c.name + "' (" + c.type + ") needs " +
                                 std::to_string(n) + " inputs, got " +
                                 std::to_string(c.inputs.size()));
}

void register_types() {
    auto& R = arandu::NodeRegistry::instance();

    R.reg("load", [](arandu::GraphBuilder& gb, const arandu::NodeConfig& c) {
        gb.add<PathItem, InputImage>(c.name, std::make_shared<LoadNode>(), c.inputs);
    });

    R.reg("fnet", [](arandu::GraphBuilder& gb, const arandu::NodeConfig& c) {
        FnetOnnxConfig mc;
        mc.path = c.get("onnx");
        if (mc.path.empty()) throw std::runtime_error("node '" + c.name + "': param onnx= is required");
        mc.provider = c.get("provider", "cuda");
        mc.device_id = run_gpu;          // policy, not a node param (see run_gpu)
        mc.max_batch = c.get_int("batch", 8);
        mc.fp16 = truthy(c.get("fp16", "0"));
        mc.engine_cache = c.get("engine_cache");
        mc.threshold = std::stof(c.get("threshold", "0.05"));
        mc.tf32 = truthy(c.get("tf32", "1"));
        mc.conv_algo = c.get("conv_algo", "HEURISTIC");
        mc.intra_threads = c.get_int("intra_threads", 2);
        // The spec may pin the shape; otherwise the probe knows it, which is the case
        // that matters -- a wrong nominal shape is a TensorRT engine rebuild per batch.
        // 0 means "probe", not 0x0: a writer that emits every field explicitly (the GUI
        // does, so the artifact says what ran) has to be able to say "unset".
        mc.nominal_h = c.get_int("nominal_h", 0);
        mc.nominal_w = c.get_int("nominal_w", 0);
        if (mc.nominal_h <= 0) mc.nominal_h = probe_shape.h;
        if (mc.nominal_w <= 0) mc.nominal_w = probe_shape.w;
        auto model = std::make_shared<FingernetOnnx>(mc);
        gb.add_model<InputImage, FnetRaw>(c.name, model, model, c.inputs);
    });

    // The postproc half, as the fan-out it is: four blocks that read only the model,
    // and the two that genuinely need the mask taking it as a second input. The chain
    // in arandu_nodes.hpp is not registered -- a graph you can draw should not offer a
    // shape whose edges claim dependencies that do not exist.
    R.reg("mask", [](arandu::GraphBuilder& gb, const arandu::NodeConfig& c) {
        gb.add<FnetRaw, fan::MaskProduct>(c.name, std::make_shared<fan::Mask>(), c.inputs);
    });
    R.reg("quality", [](arandu::GraphBuilder& gb, const arandu::NodeConfig& c) {
        gb.add<FnetRaw, fan::QualityProduct>(c.name, std::make_shared<fan::Quality>(), c.inputs);
    });
    R.reg("ori", [](arandu::GraphBuilder& gb, const arandu::NodeConfig& c) {
        gb.add<FnetRaw, fan::OriProduct>(c.name, std::make_shared<fan::Ori>(), c.inputs);
    });
    R.reg("enhanced", [](arandu::GraphBuilder& gb, const arandu::NodeConfig& c) {
        gb.add<FnetRaw, fan::EnhancedProduct>(c.name, std::make_shared<fan::Enhanced>(), c.inputs);
    });
    R.reg("enhanced_mod", [](arandu::GraphBuilder& gb, const arandu::NodeConfig& c) {
        need_inputs(c, 2);
        gb.add2<FnetRaw, fan::MaskProduct, fan::EnhancedProduct>(
            c.name, std::make_shared<fan::EnhancedMod>(), {c.inputs[0], c.inputs[1]});
    });
    R.reg("minutiae", [](arandu::GraphBuilder& gb, const arandu::NodeConfig& c) {
        need_inputs(c, 2);
        gb.add2<FnetRaw, fan::MaskProduct, fan::MinutiaeProduct>(
            c.name, std::make_shared<fan::Minutiae>(), {c.inputs[0], c.inputs[1]});
    });
    R.reg("serialize", [](arandu::GraphBuilder& gb, const arandu::NodeConfig& c) {
        need_inputs(c, 6);
        gb.addN(c.name, std::make_shared<fan::Serialize>(c.get("out", "none")), c.inputs);
    });

    // The fused postproc and its 1-input sink: same math, one phase instead of six.
    // Kept because fewer phases is measurably cheaper (a phase is a channel, a worker
    // pool and a queue hop), so the fan-out is the honest picture and this is the fast
    // path -- the editor offers both and the profiler says which one this run wanted.
    R.reg("postproc", [](arandu::GraphBuilder& gb, const arandu::NodeConfig& c) {
        auto n = std::make_shared<PostprocNode>();
        n->full = truthy(c.get("full", "0"));
        gb.add<FnetRaw, FnetProducts>(c.name, n, c.inputs);
    });
    R.reg("serialize_fused", [](arandu::GraphBuilder& gb, const arandu::NodeConfig& c) {
        gb.add<FnetProducts, Written>(c.name, std::make_shared<SerializeNode>(c.get("out", "none")),
                                      c.inputs);
    });
}

std::vector<PathItem> collect(const std::string& images) {
    std::vector<PathItem> items;
    if (images.size() > 4 && images.compare(images.size() - 4, 4, ".txt") == 0) {
        std::ifstream f(images);
        if (!f) die("cannot read list " + images);
        for (std::string line; std::getline(f, line);) {
            line = trim(line);
            if (!line.empty()) items.push_back({line, fs::path(line).stem().string()});
        }
    } else {
        if (!fs::exists(images)) die("no such path: " + images);
        if (fs::is_regular_file(images)) {
            items.push_back({images, fs::path(images).stem().string()});
            return items;
        }
        std::vector<std::string> files;
        for (auto& e : fs::recursive_directory_iterator(images))
            if (e.is_regular_file() && e.path().extension() == ".png")
                files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
        // lexically_relative, like full_graph: `relative` resolves symlinks, and the
        // dataset tree is a symlink farm, so ids would come out full of "../".
        for (auto& f : files)
            items.push_back({f, fs::path(f).lexically_relative(images).replace_extension("").string()});
    }
    return items;
}

/// How many minutiae the sink reported, for whichever sink the drawn graph ended at.
/// A graph may legitimately stop at `minutiae` (FnetProducts) instead of `serialize`
/// (Written) -- the UI decides, so the reporting cannot assume one of them.
struct SinkCount { std::string id; int n_minutiae = -1; };

SinkCount count_one(const std::any& v) {
    if (const auto* w = std::any_cast<Written>(&v)) return {w->id, w->n_minutiae};
    if (const auto* p = std::any_cast<FnetProducts>(&v))
        return {p->raw ? p->raw->id : std::string(),
                p->minutiae ? static_cast<int>(p->minutiae->size()) : -1};
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) die("usage: graph_run <spec-file> [--profile]");
    const bool want_profile = argc > 2 && std::string(argv[2]) == "--profile";
    const Spec spec = read_spec(argv[1]);

    const std::string images = spec.get("images");
    if (images.empty()) die("spec needs an @images line");
    std::vector<PathItem> items = collect(images);
    const int limit = spec.get_int("n", 0);
    if (limit > 0 && static_cast<int>(items.size()) > limit) items.resize(limit);
    if (items.empty()) die("no images under " + images);

    // The padded shape, before any session exists (see ProbeShape).
    {
        InputImage probe;
        arandu::RunCtx pc;
        LoadNode{}.run(std::span<const PathItem>(&items[0], 1), std::span<InputImage>(&probe, 1), pc);
        probe_shape = {probe.H, probe.W};
    }

    // BEFORE build(): the model node's factory reads `run_gpu` to construct the session.
    run_gpu = spec.get_int("gpu", 0);

    register_types();
    const auto t0 = clk::now();
    arandu::Graph g;
    try {
        g = arandu::NodeRegistry::instance().build(spec.nodes);
    } catch (const std::exception& exc) {
        // The graph the user drew is invalid, and arandu said exactly why (unknown type,
        // type mismatch on edge, cycle, missing sink). Passing that through verbatim is
        // the whole point of letting build() be the validator.
        std::fprintf(stderr, "graph_run: invalid graph: %s\n", exc.what());
        std::printf("{\"ok\":false,\"error\":\"%s\"}\n", esc(exc.what()).c_str());
        return 3;
    }
    const double t_build = secs(t0, clk::now());

    arandu::Profiler prof;
    arandu::ExecPolicy pol;
    pol.determinism = spec.get("determinism", "tolerant") == "bitexact"
                          ? arandu::Determinism::BitExact : arandu::Determinism::Tolerant;
    pol.device = spec.get("device", "cuda") == "cpu" ? arandu::Device::Cpu : arandu::Device::Cuda;
    pol.gpu_id = run_gpu;               // o mesmo número que a sessão ONNX recebeu
    pol.cpu_threads = spec.get_int("threads", 8);
    pol.max_batch = spec.get_int("batch", 8);
    pol.channel_depth = spec.get_int("depth", 4);
    // Four conditions gate a model phase onto its IModel rather than its cpu fallback,
    // and gpu_streams is the one that defaults to 0 -- a spec that forgets it gets the
    // fallback silently (7% slower, nothing complains). Defaulted to 1 here for that
    // reason; `@streams 0` still forces the fallback on purpose.
    pol.gpu_streams = spec.get_int("streams", 1);
    pol.model_actors = spec.get_int("actors", 2);
    pol.profiler = &prof;

    // Which implementation each phase resolved to, decided BEFORE the run and reported
    // whatever happens. A model phase falls back to its cpu kernel unless four things
    // hold at once (model impl present, device=Cuda, gpu_streams>0, Tolerant), and the
    // fallback is silent: same CUDA session, busy GPU, ~7% slower, no complaint. A UI
    // that offers "GPU or CPU" has to show which one it actually got.
    std::string phases;
    for (const auto& p : g.phases) {
        const auto impl = arandu::resolve(p, pol);
        if (!phases.empty()) phases += ",";
        phases += "{\"name\":\"" + esc(p.name) + "\",\"impl\":\"" +
                  (impl == arandu::PhaseImpl::Model ? "model"
                   : impl == arandu::PhaseImpl::Join ? "join" : "cpu") + "\"}";
    }

    const auto t1 = clk::now();
    std::vector<std::any> input(std::make_move_iterator(items.begin()),
                                std::make_move_iterator(items.end()));
    std::vector<std::any> out;
    try {
        if (spec.get("exec", "pipeline") == "serial")
            out = arandu::SerialExecutor{pol}.run(g, input);
        else
            out = arandu::PipelineExecutor{pol}.run(g, input);
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "graph_run: run failed: %s\n", exc.what());
        std::printf("{\"ok\":false,\"error\":\"%s\"}\n", esc(exc.what()).c_str());
        return 4;
    }
    const double wall = secs(t1, clk::now());

    long total_minutiae = 0;
    int counted = 0;
    std::string per_item;
    for (const auto& v : out) {
        const SinkCount c = count_one(v);
        if (c.n_minutiae < 0) continue;
        total_minutiae += c.n_minutiae;
        ++counted;
        if (!per_item.empty()) per_item += ",";
        per_item += "{\"id\":\"" + esc(c.id) + "\",\"n_minutiae\":" +
                    std::to_string(c.n_minutiae) + "}";
    }

    if (want_profile) std::fprintf(stderr, "%s\n", prof.markdown().c_str());

    std::printf("{\"ok\":true,\"images\":%zu,\"completed\":%zu,\"counted\":%d,"
                "\"minutiae\":%ld,\"wall_s\":%.3f,\"img_per_s\":%.3f,\"build_s\":%.3f,"
                "\"sink\":\"%s\",\"phases\":[%s],\"items\":[%s]}\n",
                out.size(), out.size(), counted, total_minutiae, wall,
                wall > 0 ? out.size() / wall : 0.0, t_build, esc(g.sink).c_str(),
                phases.c_str(), per_item.c_str());
    return 0;
}
