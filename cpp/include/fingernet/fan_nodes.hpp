// The postproc blocks wired as what they ARE: a fan-out from the model, not a chain.
//
// arandu_nodes.hpp threads one accumulating carrier through five nodes in a line. That
// line is not the data flow -- it is the order the nodes were written. Reading the
// kernels: `quality` reads `segmentation`, `ori` reads `orientation_index`, `enhanced`
// reads `enhanced_real`; none of the three touches anything `mask` produces. Only
// `minutiae` genuinely needs the mask (it gates the score with `cleaned` BEFORE the
// NMS, which is load-bearing), and only the `_mod` variant of enhanced does.
//
// So the honest graph is:
//
//     fnet ─┬─▶ mask ──────┬─────────────┐
//           ├─▶ quality    │             │
//           ├─▶ ori        │             ├─▶ serialize
//           ├─▶ enhanced   │             │
//           └──────────────┴─▶ minutiae ─┘        (minutiae JOINs raw + mask)
//
// Every node emits ITS OWN product type, so a mis-wire is a build() error instead of a
// null field at run time, and the picture on screen is the dependency graph.
//
// ADDITIVE on purpose: arandu_nodes.hpp keeps the chain and the fused PostprocNode,
// which the parity gate and the shipped drivers use. This set is what the graph editor
// offers; the chain retires once this one has carried a full run.
#pragma once
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "arandu/kernel.hpp"
#include "arandu_nodes.hpp"
#include "io_nodes.hpp"     // Written + SerializeNode's writing (and so libpng)
#include "minfmt.hpp"
#include "postproc.hpp"

namespace fnaru::fan {

// ── products ────────────────────────────────────────────────────────────────
//
// Each carries the `raw` it came from: a shared_ptr costs a refcount bump, and it is
// what lets any consumer find the dims and the id without a second wire for them.

struct MaskProduct {
    std::shared_ptr<const RawImage> raw;
    std::shared_ptr<const std::vector<float>> cleaned;             // coarse {0,1}
    std::shared_ptr<const std::vector<uint8_t>> segmentation_mask; // [H*W]
};
struct QualityProduct {
    std::shared_ptr<const std::vector<uint8_t>> quality;           // [H*W]
};
struct OriProduct {
    std::shared_ptr<const std::vector<float>> field;               // [H*W] radians
};
struct EnhancedProduct {
    std::shared_ptr<const std::vector<uint8_t>> image;             // [H*W]
    std::shared_ptr<const std::vector<uint8_t>> image_mod;         // null unless _mod
};
struct MinutiaeProduct {
    std::shared_ptr<const std::vector<fnpost::Minutia>> minutiae;
};

// ── the four independent blocks: FnetRaw in, one product out ────────────────

struct Mask : arandu::ICpuScript<FnetRaw, MaskProduct> {
    void run(std::span<const FnetRaw> in, std::span<MaskProduct> out,
             arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            const RawImage& r = *in[i].raw;
            auto cleaned = std::make_shared<std::vector<float>>(
                fnpost::binarize_mask_fast(r.segmentation.data(), r.h, r.w));
            out[i] = MaskProduct{
                in[i].raw, cleaned,
                std::make_shared<std::vector<uint8_t>>(
                    fnpost::mask_up_u8(cleaned->data(), r.h, r.w))};
        }
    }
};

struct Quality : arandu::ICpuScript<FnetRaw, QualityProduct> {
    void run(std::span<const FnetRaw> in, std::span<QualityProduct> out,
             arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            const RawImage& r = *in[i].raw;
            out[i] = QualityProduct{std::make_shared<std::vector<uint8_t>>(
                fnpost::quality_u8(r.segmentation.data(), r.h, r.w))};
        }
    }
};

struct Ori : arandu::ICpuScript<FnetRaw, OriProduct> {
    void run(std::span<const FnetRaw> in, std::span<OriProduct> out,
             arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            const RawImage& r = *in[i].raw;
            out[i] = OriProduct{std::make_shared<std::vector<float>>(
                fnpost::orientation_field(r.orientation_index.data(), r.h, r.w))};
        }
    }
};

struct Enhanced : arandu::ICpuScript<FnetRaw, EnhancedProduct> {
    void run(std::span<const FnetRaw> in, std::span<EnhancedProduct> out,
             arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            const RawImage& r = *in[i].raw;
            out[i] = EnhancedProduct{std::make_shared<std::vector<uint8_t>>(
                fnpost::enhanced_u8(r.enhanced_real.data(), r.H, r.W)), nullptr};
        }
    }
};

// The masked variant is a DIFFERENT node, not a flag: it genuinely needs the mask, and
// a flag would have made every graph carry that dependency to serve the case that uses
// it. Normalising the masked float puts the background at mid-grey, which is why it is
// not the mask times the plain product.
struct EnhancedMod : arandu::IJoin2<FnetRaw, MaskProduct, EnhancedProduct> {
    void run(std::span<const FnetRaw> raw, std::span<const MaskProduct> mask,
             std::span<EnhancedProduct> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < raw.size(); ++i) {
            const RawImage& r = *raw[i].raw;
            out[i] = EnhancedProduct{
                std::make_shared<std::vector<uint8_t>>(
                    fnpost::enhanced_u8(r.enhanced_real.data(), r.H, r.W)),
                std::make_shared<std::vector<uint8_t>>(
                    fnpost::enhanced_masked_u8(r.enhanced_real.data(),
                                               mask[i].cleaned->data(), r.h, r.w))};
        }
    }
};

// The one real dependency in the postproc half: the score is gated by the mask BEFORE
// the NMS. Masking afterwards is a different operation -- a background candidate that
// survives into an order-dependent NMS can suppress a real foreground minutia.
struct Minutiae : arandu::IJoin2<FnetRaw, MaskProduct, MinutiaeProduct> {
    void run(std::span<const FnetRaw> raw, std::span<const MaskProduct> mask,
             std::span<MinutiaeProduct> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < raw.size(); ++i)
            out[i] = MinutiaeProduct{std::make_shared<std::vector<fnpost::Minutia>>(
                fnaru::detect_minutiae(*raw[i].raw, *mask[i].cleaned))};
    }
};

// ── the sink: six inputs, because it reads six things ───────────────────────

struct Serialize : arandu::IJoinN<Written, FnetRaw, MaskProduct, QualityProduct,
                                  OriProduct, EnhancedProduct, MinutiaeProduct> {
    std::string out;
    explicit Serialize(std::string o) : out(std::move(o)) {}

    void run(std::span<const FnetRaw> raw, std::span<const MaskProduct> mask,
             std::span<const QualityProduct> quality, std::span<const OriProduct> ori,
             std::span<const EnhancedProduct> enh, std::span<const MinutiaeProduct> mnt,
             std::span<Written> outv, arandu::RunCtx&) const override {
        // The writing itself is SerializeNode's, reused rather than copied: the file
        // names, the crop and the .min conversion are contract with api.py, and a
        // second copy of them is a second thing to keep in step.
        const SerializeNode writer(out);
        for (size_t i = 0; i < raw.size(); ++i) {
            const RawImage& r = *raw[i].raw;
            if (out == "none") {
                outv[i] = Written{r.id, static_cast<int>(mnt[i].minutiae->size())};
                continue;
            }
            writer.save("enhanced", r.id, *enh[i].image, r.W, r.orig_h, r.orig_w);
            writer.save("mask", r.id, *mask[i].segmentation_mask, r.W, r.orig_h, r.orig_w);
            writer.save("quality", r.id, *quality[i].quality, r.W, r.orig_h, r.orig_w);
            writer.save("ori", r.id,
                        fnpost::orientation_png(r.orientation_index.data(), r.h, r.w),
                        r.W, r.orig_h, r.orig_w);
            writer.save_min("minutiae", r.id, *mnt[i].minutiae);
            if (enh[i].image_mod) {
                writer.save("enhanced_mod", r.id, *enh[i].image_mod, r.W, r.orig_h, r.orig_w);
                writer.save("ori_mod", r.id,
                            fnpost::orientation_png(r.orientation_index.data(), r.h, r.w,
                                                    mask[i].cleaned->data()),
                            r.W, r.orig_h, r.orig_w);
            }
            outv[i] = Written{r.id, static_cast<int>(mnt[i].minutiae->size())};
        }
        // `ori` is read for its field only when a consumer wants it; the PNG is
        // quantised from the index plane (64x fewer lround calls), same as the chain.
        (void)ori;
    }
};

}  // namespace fnaru::fan
