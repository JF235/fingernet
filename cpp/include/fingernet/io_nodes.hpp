// I/O arandu nodes (no ONNX dependency): LoadNode reads a PNG into InputImage;
// SerializeNode writes the production artifacts (PNG + .min). Kept separate so both
// the full graph and the stage benchmarks can share them.
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

// serialize(sink): crop maps to orig, write the artifacts api.py writes -- the five
// defaults (minutiae/ enhanced/ mask/ ori/ quality/), plus enhanced_mod/ and ori_mod/
// when the producer made them. Same set, same names, same directory layout, so the two extractions
// diff directly.
// The _mod pair is written iff the producer made it: there is no second flag here to
// disagree with the producer's, which is how a null deref used to get past build().
struct SerializeNode : arandu::ICpuScript<FnetProducts, Written> {
    std::string out;
    // zlib effort for every PNG this writes. Not libpng's 6: that level was 76 ms of this
    // node's 82 ms/item, and PNG being lossless it changes the file size, never a pixel.
    int png_level;
    explicit SerializeNode(std::string o, int level = fnpng::kDefaultLevel)
        : out(std::move(o)), png_level(level) {}
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
        fnpng::write_gray(p.string(), c.data(), oh, ow, png_level);
    }
    void save_min(const std::string& sub, const std::string& id,
                  const std::vector<fnpost::Minutia>& m) const {
        namespace fs = std::filesystem;
        fs::path p = fs::path(out) / sub / (id + ".min");
        fs::create_directories(p.parent_path());
        fnmin::write_min(p.string(), fnmin::to_min_rows(m));
    }
    void run(std::span<const FnetProducts> in, std::span<Written> outv, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            const FnetProducts& b = in[i]; const RawImage& r = *b.raw;
            if (out == "none") { outv[i] = Written{r.id, (int)b.minutiae->size()}; continue; }  // I/O-isolation
            int oh = r.orig_h, ow = r.orig_w, W = r.W;
            // Quantised from the coarse index plane, not from the [H,W] float field:
            // both derive from the same bins through the same expression, and deciding
            // the byte per cell is 64x fewer lround calls (5.3 -> 0.5 ms/image).
            save("enhanced", r.id, *b.enhanced_image, W, oh, ow);
            save("mask", r.id, *b.segmentation_mask, W, oh, ow);
            save("quality", r.id, *b.quality, W, oh, ow);
            save("ori", r.id, fnpost::orientation_png(r.orientation_index.data(), r.h, r.w), W, oh, ow);
            save_min("minutiae", r.id, *b.minutiae);
            if (b.enhanced_image_mod) {
                // ori_mod is the mask times a primitive, so it is derived here rather
                // than carried; enhanced_mod is not (see enhanced_masked_u8).
                save("enhanced_mod", r.id, *b.enhanced_image_mod, W, oh, ow);
                save("ori_mod", r.id,
                     fnpost::orientation_png(r.orientation_index.data(), r.h, r.w, b.cleaned->data()),
                     W, oh, ow);
            }
            outv[i] = Written{r.id, (int)b.minutiae->size()};
        }
    }
};

}  // namespace fnaru
