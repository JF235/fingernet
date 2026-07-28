// Full serialization to disk: run the postproc blocks and write the production
// artifacts (enhanced/mask/quality/ori PNGs + minutiae .min) for N images.
// Usage: serialize_demo <refdump_dir> <out_dir> [num_imgs]
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "fingernet/minfmt.hpp"
#include "fingernet/npy.hpp"
#include "fingernet/png.hpp"
#include "fingernet/postproc.hpp"

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

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "/storage/jcontreras/tmp_arandu_bench/refdump516";
    std::string out = argc > 2 ? argv[2] : "./cpp_out";
    int N = argc > 3 ? std::atoi(argv[3]) : 8;
    int h = meta_int(dir, "h", 96), w = meta_int(dir, "w", 100);
    int H = h * 8, W = w * 8, hw = h * w, HW = H * W;
    float thr = meta_f(dir, "threshold", 0.05f);
    for (auto s : {"enhanced", "mask", "quality", "ori", "minutiae"})
        system(("mkdir -p " + out + "/" + s).c_str());

    auto R = [&](const char* p, int i) { return dir + "/raw/" + p + "_" + std::to_string(i) + ".npy"; };
    for (int i = 0; i < N; ++i) {
        fnpy::Array seg = fnpy::load(R("segmentation", i));
        fnpy::Array ori = fnpy::load(R("orientation", i));
        fnpy::Array enh = fnpy::load(R("enhanced_real", i));
        fnpy::Array mo = fnpy::load(R("minutiae_orientation", i));
        fnpy::Array mx = fnpy::load(R("minutiae_x_offset", i));
        fnpy::Array my = fnpy::load(R("minutiae_y_offset", i));
        fnpy::Array ms = fnpy::load(R("minutiae_score", i));

        auto cleaned = fnpost::binarize_mask_fast(seg.as<float>(), h, w);
        auto mask = fnpost::mask_up_u8(cleaned.data(), h, w);
        auto qual = fnpost::quality_u8(seg.as<float>(), h, w);
        auto orif = fnpost::orientation_field(ori.as<float>(), 90, h, w);
        std::vector<uint8_t> orip(HW);
        for (int k = 0; k < HW; ++k) orip[k] = (uint8_t)std::lround(orif[k] * 180.0 / fnpost::PI + 90.0);
        auto en = fnpost::enhanced_u8(enh.as<float>(), H, W);
        std::vector<float> masked(hw);
        for (int k = 0; k < hw; ++k) masked[k] = ms.as<float>()[k] * cleaned[k];
        auto mnt = fnpost::detect_minutiae(masked.data(), mo.as<float>(), mx.as<float>(), my.as<float>(), h, w, thr);

        std::string id = std::to_string(i);
        fnpng::write_gray(out + "/enhanced/" + id + ".png", en.data(), H, W);
        fnpng::write_gray(out + "/mask/" + id + ".png", mask.data(), H, W);
        fnpng::write_gray(out + "/quality/" + id + ".png", qual.data(), H, W);
        fnpng::write_gray(out + "/ori/" + id + ".png", orip.data(), H, W);
        fnmin::write_min(out + "/minutiae/" + id + ".min", fnmin::to_min_rows(mnt));
    }
    printf("wrote %d imgs (enhanced/mask/quality/ori PNG + minutiae .min) to %s\n", N, out.c_str());
    return 0;
}
