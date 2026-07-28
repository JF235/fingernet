// Per-block parity: C++ postproc vs the Python (PyTorch, CPU) reference dump.
// Per-image .npy layout (see dump_reference_stream.py). Usage: parity <dir> [write_min]
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "fingernet/minfmt.hpp"
#include "fingernet/npy.hpp"
#include "fingernet/postproc.hpp"

using fnpy::Array;

static int meta_int(const std::string& dir, const std::string& key, int def) {
    std::ifstream f(dir + "/meta.txt");
    std::string k;
    while (f >> k) {
        std::string rest;
        if (k == key) { int v; f >> v; return v; }
        std::getline(f, rest);
    }
    return def;
}
static float meta_f(const std::string& dir, const std::string& key, float def) {
    std::ifstream f(dir + "/meta.txt");
    std::string k;
    while (f >> k) {
        std::string rest;
        if (k == key) { float v; f >> v; return v; }
        std::getline(f, rest);
    }
    return def;
}

struct FStat { double maxabs = 0; long over = 0; long n = 0; };
static void cmp_f(const std::vector<float>& a, const float* b, double tol, FStat& s) {
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        s.maxabs = std::max(s.maxabs, d);
        if (d > tol) ++s.over;
    }
    s.n += (long)a.size();
}
struct U8Stat { int maxabs = 0; long mism = 0; long n = 0; long imgs_exact = 0; };
static void cmp_u8(const std::vector<uint8_t>& a, const uint8_t* b, U8Stat& s) {
    long m0 = s.mism;
    for (size_t i = 0; i < a.size(); ++i) {
        int d = std::abs((int)a[i] - (int)b[i]);
        s.maxabs = std::max(s.maxabs, d);
        if (d != 0) ++s.mism;
    }
    s.n += (long)a.size();
    if (s.mism == m0) ++s.imgs_exact;
}

static std::vector<std::array<long,4>> read_min_txt(const std::string& path) {
    std::vector<std::array<long,4>> rows;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::array<long,4> r{};
        if (ss >> r[0] >> r[1] >> r[2] >> r[3]) rows.push_back(r);
    }
    return rows;
}
static void cmp_min(const std::vector<fnmin::MinRow>& cpp,
                    const std::vector<std::array<long,4>>& ref, long& exact) {
    std::multiset<std::array<long,4>> R;
    for (auto& r : ref) R.insert(r);
    for (auto& m : cpp) {
        auto it = R.find({m.x, m.y, m.angle, m.quality});
        if (it != R.end()) { ++exact; R.erase(it); }
    }
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "./refdump";
    bool write_min = argc > 2 && std::string(argv[2]) == "write_min";
    int N = meta_int(dir, "N", meta_int(dir, "B", 8));
    int h = meta_int(dir, "h", 96), w = meta_int(dir, "w", 100);
    float threshold = meta_f(dir, "threshold", 0.05f);
    int H = h * 8, W = w * 8, hw = h * w, HW = H * W;
    printf("N=%d  coarse=%dx%d  full=%dx%d  threshold=%.3f\n", N, h, w, H, W, threshold);
    if (write_min) system(("mkdir -p " + dir + "/cpp_min").c_str());

    FStat A_clean{}, C_orif{};
    U8Stat A_mask{}, B_qual{}, C_orip{}, D_enh{}, F_enhm{};
    long e_exact=0,e_ref=0,e_cpp=0, u_exact=0,u_ref=0,u_cpp=0;
    int worst_q_img = -1; long worst_q = -1;

    auto R = [&](const char* pre, int i) { return dir + "/raw/" + pre + "_" + std::to_string(i) + ".npy"; };
    auto F = [&](const char* pre, int i) { return dir + "/ref/" + pre + "_" + std::to_string(i) + ".npy"; };

    for (int i = 0; i < N; ++i) {
        Array seg = fnpy::load(R("segmentation", i));           // [1,h,w]
        Array ori = fnpy::load(R("orientation", i));            // [90,h,w]
        Array enh = fnpy::load(R("enhanced_real", i));          // [1,H,W]
        Array mo  = fnpy::load(R("minutiae_orientation", i));   // [180,h,w]
        Array mx  = fnpy::load(R("minutiae_x_offset", i));      // [8,h,w]
        Array my  = fnpy::load(R("minutiae_y_offset", i));      // [8,h,w]
        Array ms  = fnpy::load(R("minutiae_score", i));         // [1,h,w]

        const float* seg_i = seg.as<float>();
        // A
        auto cleaned = fnpost::binarize_mask_fast(seg_i, h, w);
        { Array r = fnpy::load(F("cleaned_mask", i)); cmp_f(cleaned, r.as<float>(), 0.0, A_clean); }
        auto mask = fnpost::mask_up_u8(cleaned.data(), h, w);
        { Array r = fnpy::load(F("segmentation_mask", i)); cmp_u8(mask, r.as<uint8_t>(), A_mask); }
        // B
        auto qual = fnpost::quality_u8(seg_i, h, w);
        { Array r = fnpy::load(F("quality", i)); long m0=B_qual.mism; cmp_u8(qual, r.as<uint8_t>(), B_qual);
          if (B_qual.mism-m0 > worst_q) { worst_q = B_qual.mism-m0; worst_q_img = i; } }
        // C
        auto orif = fnpost::orientation_field(ori.as<float>(), 90, h, w);
        std::vector<uint8_t> orip(HW);
        for (int k = 0; k < HW; ++k) orip[k] = (uint8_t)std::lround(orif[k]*180.0/fnpost::PI + 90.0);
        { Array r = fnpy::load(F("ori_png", i)); cmp_u8(orip, r.as<uint8_t>(), C_orip); }
        // D
        auto en = fnpost::enhanced_u8(enh.as<float>(), H, W);
        { Array r = fnpy::load(F("enhanced_image", i)); cmp_u8(en, r.as<uint8_t>(), D_enh); }
        // F2
        auto cleaned_up = fnpost::nearest_up(cleaned.data(), h, w);
        std::vector<float> enh_mod(HW);
        for (int k = 0; k < HW; ++k) enh_mod[k] = enh.as<float>()[k] * cleaned_up[k];
        auto enm = fnpost::enhanced_u8(enh_mod.data(), H, W);
        { Array r = fnpy::load(F("enhanced_image_mod", i)); cmp_u8(enm, r.as<uint8_t>(), F_enhm); }
        // E
        std::vector<float> masked(hw);
        for (int k = 0; k < hw; ++k) masked[k] = ms.as<float>()[k] * cleaned[k];
        auto mnt = fnpost::detect_minutiae(masked.data(), mo.as<float>(), mx.as<float>(), my.as<float>(), h, w, threshold);
        auto rows = fnmin::to_min_rows(mnt);
        if (write_min) fnmin::write_min(dir + "/cpp_min/min_" + std::to_string(i) + ".min", rows);
        { auto ref = read_min_txt(dir + "/ref/min/min_" + std::to_string(i) + ".txt");
          cmp_min(rows, ref, e_exact); e_ref += (long)ref.size(); e_cpp += (long)rows.size(); }
        auto mnt_u = fnpost::detect_minutiae(ms.as<float>(), mo.as<float>(), mx.as<float>(), my.as<float>(), h, w, threshold);
        auto rows_u = fnmin::to_min_rows(mnt_u);
        { auto ref = read_min_txt(dir + "/ref/min/min_unmod_" + std::to_string(i) + ".txt");
          cmp_min(rows_u, ref, u_exact); u_ref += (long)ref.size(); u_cpp += (long)rows_u.size(); }

        if ((i+1) % 64 == 0) { printf("\r  %d/%d", i+1, N); fflush(stdout); }
    }

    auto pct = [](long a, long b) { return b ? 100.0 * a / b : 100.0; };
    printf("\n== blocks vs Python (all %d imgs) ==\n", N);
    printf("A cleaned_mask (coarse f32): max|d|=%.3g  mismatches=%ld/%ld\n", A_clean.maxabs, A_clean.over, A_clean.n);
    printf("A segmentation_mask  (u8) : max|d|=%d  mism=%ld/%ld  imgs_exact=%ld/%d\n", A_mask.maxabs, A_mask.mism, A_mask.n, A_mask.imgs_exact, N);
    printf("B quality            (u8) : max|d|=%d  mism=%ld/%ld  imgs_exact=%ld/%d  (worst img %d: %ld px)\n", B_qual.maxabs, B_qual.mism, B_qual.n, B_qual.imgs_exact, N, worst_q_img, worst_q);
    printf("C ori_png            (u8) : max|d|=%d  mism=%ld/%ld  imgs_exact=%ld/%d\n", C_orip.maxabs, C_orip.mism, C_orip.n, C_orip.imgs_exact, N);
    printf("D enhanced_image     (u8) : max|d|=%d  mism=%ld/%ld  imgs_exact=%ld/%d\n", D_enh.maxabs, D_enh.mism, D_enh.n, D_enh.imgs_exact, N);
    printf("F enhanced_image_mod (u8) : max|d|=%d  mism=%ld/%ld  imgs_exact=%ld/%d\n", F_enhm.maxabs, F_enhm.mism, F_enhm.n, F_enhm.imgs_exact, N);
    printf("E minutiae      : exact=%ld/%ld (%.4f%%)  [cpp=%ld ref=%ld]\n", e_exact, e_ref, pct(e_exact, e_ref), e_cpp, e_ref);
    printf("E minutiae_unmod: exact=%ld/%ld (%.4f%%)  [cpp=%ld ref=%ld]\n", u_exact, u_ref, pct(u_exact, u_ref), u_cpp, u_ref);
    return 0;
}
