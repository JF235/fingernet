// I/O arandu nodes (no ONNX dependency): LoadNode reads a PNG into InputImage;
// SerializeNode writes the 8 production artifacts (PNG + .min). Kept separate
// so both the full graph and stage benchmarks can share them.
#pragma once
#include <cmath>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "arandu/kernel.hpp"
#include "arandu_nodes.hpp"
#include "minfmt.hpp"
#include "png.hpp"
#include "postproc.hpp"

namespace fnaru {

struct PathItem { std::string path, id; };
struct Written { std::string id; int n_minutiae = 0; };

// load(source): read PNG -> grayscale/255 -> pad to /8 -> InputImage
struct LoadNode : arandu::ICpuScript<PathItem, InputImage> {
    void run(std::span<const PathItem> in, std::span<InputImage> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            int H0 = 0, W0 = 0;
            auto px = fnpng::read_gray(in[i].path, H0, W0);
            int H = H0 + (8 - H0 % 8) % 8, W = W0 + (8 - W0 % 8) % 8;
            InputImage im;
            im.orig_h = H0; im.orig_w = W0; im.H = H; im.W = W; im.h = H / 8; im.w = W / 8; im.id = in[i].id;
            im.data.assign(static_cast<size_t>(H) * W, 0.0f);
            for (int y = 0; y < H0; ++y)
                for (int x = 0; x < W0; ++x)
                    im.data[static_cast<size_t>(y) * W + x] = px[static_cast<size_t>(y) * W0 + x] / 255.0f;
            out[i] = std::move(im);
        }
    }
};

// serialize(sink): crop maps to orig, write the 8 --full artifacts.
struct SerializeNode : arandu::ICpuScript<Bundle, Written> {
    std::string out;
    explicit SerializeNode(std::string o) : out(std::move(o)) {}
    static std::vector<uint8_t> crop(const std::vector<uint8_t>& s, int W, int oh, int ow) {
        std::vector<uint8_t> d(static_cast<size_t>(oh) * ow);
        for (int y = 0; y < oh; ++y)
            for (int x = 0; x < ow; ++x) d[static_cast<size_t>(y) * ow + x] = s[static_cast<size_t>(y) * W + x];
        return d;
    }
    void save(const std::string& sub, const std::string& id, const std::vector<uint8_t>& px,
              int W, int oh, int ow) const {
        namespace fs = std::filesystem;
        fs::path p = fs::path(out) / sub / (id + ".png");
        fs::create_directories(p.parent_path());
        auto c = (oh == (int)(px.size() / W) && ow == W) ? px : crop(px, W, oh, ow);
        fnpng::write_gray(p.string(), c.data(), oh, ow);
    }
    void save_min(const std::string& sub, const std::string& id,
                  const std::vector<fnpost::Minutia>& m) const {
        namespace fs = std::filesystem;
        fs::path p = fs::path(out) / sub / (id + ".min");
        fs::create_directories(p.parent_path());
        fnmin::write_min(p.string(), fnmin::to_min_rows(m));
    }
    void run(std::span<const Bundle> in, std::span<Written> outv, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            const Bundle& b = in[i]; const RawImage& r = *b.raw;
            if (out == "none") { outv[i] = Written{r.id, (int)b.minutiae->size()}; continue; }  // I/O-isolation
            int oh = r.orig_h, ow = r.orig_w, W = r.W;
            const auto& of = *b.orientation_field;
            auto cup = fnpost::nearest_up(b.cleaned->data(), r.h, r.w);
            std::vector<uint8_t> orip(of.size()), orip_mod(of.size());
            for (size_t k = 0; k < of.size(); ++k) {
                orip[k] = (uint8_t)std::lround(of[k] * 180.0 / fnpost::PI + 90.0);
                orip_mod[k] = (uint8_t)std::lround(of[k] * cup[k] * 180.0 / fnpost::PI + 90.0);
            }
            save("enhanced", r.id, *b.enhanced_image, W, oh, ow);
            save("enhanced_mod", r.id, *b.enhanced_image_mod, W, oh, ow);
            save("mask", r.id, *b.segmentation_mask, W, oh, ow);
            save("quality", r.id, *b.quality, W, oh, ow);
            save("ori", r.id, orip, W, oh, ow);
            save("ori_mod", r.id, orip_mod, W, oh, ow);
            save_min("minutiae", r.id, *b.minutiae);
            save_min("minutiae_unmod", r.id, *b.minutiae_unmod);
            outv[i] = Written{r.id, (int)b.minutiae->size()};
        }
    }
};

}  // namespace fnaru
