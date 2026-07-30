// Demonstrate the FingerNet postproc blocks running as an arandu graph:
//  - build the chain mask->quality->ori->enhanced->minutiae
//  - SerialExecutor (ELFT floor) vs PipelineExecutor(T) must be BYTE-IDENTICAL
//  - Serial output must match the Python reference (same as the standalone gate)
// Usage: arandu_demo <refdump_dir> [num_imgs] [threads]
#include <algorithm>
#include <any>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "fingernet/npy.hpp"       // the Python reference dump; not part of the pipeline
#include "fingernet/pipeline.hpp"

using namespace fnaru;

static int meta_int(const std::string& d, const std::string& k, int def) {
    std::ifstream f(d + "/meta.txt"); std::string s;
    while (f >> s) { std::string r; if (s == k) { int v; f >> v; return v; } std::getline(f, r); }
    return def;
}
static float meta_f(const std::string& d, const std::string& k, float def) {
    std::ifstream f(d + "/meta.txt"); std::string s;
    while (f >> s) { std::string r; if (s == k) { float v; f >> v; return v; } std::getline(f, r); }
    return def;
}
static std::vector<float> load_vec(const std::string& p) {
    fnpy::Array a = fnpy::load(p);
    return std::vector<float>(a.as<float>(), a.as<float>() + a.size());
}
static std::vector<std::array<long,4>> read_min(const std::string& p) {
    std::vector<std::array<long,4>> v; std::ifstream f(p); std::string ln;
    while (std::getline(f, ln)) { if (ln.empty()||ln[0]=='#') continue; std::istringstream ss(ln);
        std::array<long,4> r{}; if (ss>>r[0]>>r[1]>>r[2]>>r[3]) v.push_back(r); }
    return v;
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "/storage/jcontreras/tmp_arandu_bench/refdump516";
    int N = argc > 2 ? std::atoi(argv[2]) : 64;
    int T = argc > 3 ? std::atoi(argv[3]) : 8;
    int h = meta_int(dir, "h", 96), w = meta_int(dir, "w", 100);
    int H = h * 8, W = w * 8;
    float thr = meta_f(dir, "threshold", 0.05f);
    N = std::min(N, meta_int(dir, "N", N));
    printf("arandu postproc graph: %d imgs, chain mask->quality->ori->enhanced->minutiae\n", N);

    // build input Bundles
    std::vector<std::any> input;
    input.reserve(N);
    auto R = [&](const char* p, int i) { return dir + "/raw/" + p + "_" + std::to_string(i) + ".npy"; };
    for (int i = 0; i < N; ++i) {
        auto r = std::make_shared<RawImage>();
        r->h = h; r->w = w; r->H = H; r->W = W; r->threshold = thr;
        // The dump predates the argmax moving into the ONNX graph, so it still holds
        // the float channel stacks; reduce them the way the graph now does.
        auto amax = [&](const char* p, int C) {
            return fnpost::argmax_plane(load_vec(R(p, i)).data(), C, h, w);
        };
        r->segmentation = load_vec(R("segmentation", i));
        r->enhanced_real = load_vec(R("enhanced_real", i));
        r->minutiae_score = load_vec(R("minutiae_score", i));
        r->orientation_index = amax("orientation", 90);
        r->minutiae_orientation_index = amax("minutiae_orientation", 180);
        r->minutiae_x_index = amax("minutiae_x_offset", 8);
        r->minutiae_y_index = amax("minutiae_y_offset", 8);
        Bundle b; b.raw = r;
        input.emplace_back(std::move(b));
    }

    arandu::Graph g = build_postproc_graph();

    using clk = std::chrono::steady_clock;
    arandu::SerialExecutor serial;
    auto t0 = clk::now();
    auto rs = serial.run(g, input);
    auto t1 = clk::now();
    arandu::ExecPolicy pol = arandu::ExecPolicy::elft_submission(); pol.cpu_threads = T;
    // Was StreamExecutor, which arandu retired into the PipelineExecutor (it was the
    // same thing minus the model phases). On this CPU-only graph under elft_submission
    // every phase resolves to its Cpu impl, so the byte-identity asserted below is the
    // same claim it always was.
    arandu::PipelineExecutor stream(pol);
    auto t2 = clk::now();
    auto rp = stream.run(g, input);
    auto t3 = clk::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    printf("Serial: %.1f ms (%.1f img/s)   Pipeline(T=%d): %.1f ms (%.1f img/s)  speedup %.1fx\n",
           ms(t0, t1), 1000.0 * N / ms(t0, t1), T, ms(t2, t3), 1000.0 * N / ms(t2, t3), ms(t0, t1) / ms(t2, t3));

    // (1) Serial == Pipeline, byte-for-byte (thread invariance)
    long dmask = 0, dqual = 0, dori = 0, denh = 0, dmnt = 0;
    for (int i = 0; i < N; ++i) {
        const Bundle& a = std::any_cast<const Bundle&>(rs[i]);
        const Bundle& b = std::any_cast<const Bundle&>(rp[i]);
        if (*a.segmentation_mask != *b.segmentation_mask) ++dmask;
        if (*a.quality != *b.quality) ++dqual;
        if (*a.orientation_field != *b.orientation_field) ++dori;
        if (*a.enhanced_image != *b.enhanced_image) ++denh;
        if (a.minutiae->size() != b.minutiae->size()) { ++dmnt; continue; }
        for (size_t k = 0; k < a.minutiae->size(); ++k) {
            const auto& m1 = (*a.minutiae)[k]; const auto& m2 = (*b.minutiae)[k];
            if (m1.x != m2.x || m1.y != m2.y || m1.angle != m2.angle || m1.score != m2.score) { ++dmnt; break; }
        }
    }
    printf("\n[thread invariance] Serial vs Stream(T=%d) differing images: mask=%ld quality=%ld ori=%ld enhanced=%ld minutiae=%ld  -> %s\n",
           T, dmask, dqual, dori, denh, dmnt,
           (dmask+dqual+dori+denh+dmnt == 0) ? "BYTE-IDENTICAL" : "MISMATCH");

    // (2) Serial vs Python reference (minutiae exact + map mismatches)
    long e_ok = 0, e_ref = 0, e_cpp = 0, q_mism = 0, en_mism = 0, mk_mism = 0;
    auto F = [&](const char* p, int i) { return dir + "/ref/" + p + "_" + std::to_string(i) + ".npy"; };
    for (int i = 0; i < N; ++i) {
        const Bundle& a = std::any_cast<const Bundle&>(rs[i]);
        { fnpy::Array r = fnpy::load(F("quality", i)); for (size_t k=0;k<a.quality->size();++k) if ((*a.quality)[k]!=r.as<uint8_t>()[k]) ++q_mism; }
        { fnpy::Array r = fnpy::load(F("enhanced_image", i)); for (size_t k=0;k<a.enhanced_image->size();++k) if ((*a.enhanced_image)[k]!=r.as<uint8_t>()[k]) ++en_mism; }
        { fnpy::Array r = fnpy::load(F("segmentation_mask", i)); for (size_t k=0;k<a.segmentation_mask->size();++k) if ((*a.segmentation_mask)[k]!=r.as<uint8_t>()[k]) ++mk_mism; }
        auto rows = fnmin::to_min_rows(*a.minutiae);
        auto ref = read_min(dir + "/ref/min/min_" + std::to_string(i) + ".txt");
        std::multiset<std::array<long,4>> Rm(ref.begin(), ref.end());
        for (auto& m : rows) { auto it = Rm.find({m.x,m.y,m.angle,m.quality}); if (it!=Rm.end()){++e_ok;Rm.erase(it);} }
        e_ref += (long)ref.size(); e_cpp += (long)rows.size();
    }
    printf("[vs Python] minutiae exact=%ld/%ld (%.4f%%)  quality px mism=%ld  enhanced px mism=%ld  mask px mism=%ld\n",
           e_ok, e_ref, e_ref ? 100.0 * e_ok / e_ref : 100.0, q_mism, en_mism, mk_mism);
    return 0;
}
