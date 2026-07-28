// FingerNet postproc blocks wrapped as arandu nodes (ICpuScript<Bundle,Bundle>),
// item-pure (out[i] depends only on in[i]). The blocks are chained; each node
// adds its output to a shared-ptr Bundle carrier so chaining copies stay cheap.
// This is the plug-and-play adapter: arandu builds the graph, fingernet owns
// the kernels. Requires the arandu headers on the include path.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "arandu/graph.hpp"
#include "arandu/kernel.hpp"
#include "postproc.hpp"

namespace fnaru {

// The 7 raw model outputs for ONE image (single-channel dims squeezed away).
struct RawImage {
    int h = 0, w = 0, H = 0, W = 0;
    int orig_h = 0, orig_w = 0;   // pre-pad dims (output cropping)
    std::string id;               // relative id for output filenames
    float threshold = 0.05f;
    std::vector<float> segmentation;          // [h*w]
    std::vector<float> orientation;           // [90*h*w]
    std::vector<float> enhanced_real;         // [H*W]
    std::vector<float> minutiae_orientation;  // [180*h*w]
    std::vector<float> minutiae_x_offset;     // [8*h*w]
    std::vector<float> minutiae_y_offset;     // [8*h*w]
    std::vector<float> minutiae_score;        // [h*w]
};

// A preprocessed input image for the model: [H*W] grayscale/255, padded to /8.
struct InputImage {
    int h = 0, w = 0, H = 0, W = 0;   // coarse (H/8) + full dims after padding
    int orig_h = 0, orig_w = 0;       // pre-pad dims (for output cropping)
    std::string id;                   // relative id for output filenames
    std::vector<float> data;          // [H*W]
};

// Carrier threaded through the postproc chain; shared_ptr => cheap to copy.
struct Bundle {
    std::shared_ptr<const RawImage> raw;
    std::shared_ptr<const std::vector<float>> cleaned;               // coarse {0,1}
    std::shared_ptr<const std::vector<uint8_t>> segmentation_mask;   // [H*W]
    std::shared_ptr<const std::vector<uint8_t>> quality;             // [H*W]
    std::shared_ptr<const std::vector<float>> orientation_field;     // [H*W] rad
    std::shared_ptr<const std::vector<uint8_t>> enhanced_image;      // [H*W]
    std::shared_ptr<const std::vector<uint8_t>> enhanced_image_mod;  // [H*W]
    std::shared_ptr<const std::vector<fnpost::Minutia>> minutiae;
    std::shared_ptr<const std::vector<fnpost::Minutia>> minutiae_unmod;
};

// A: segmentation -> cleaned (coarse) + segmentation_mask (u8). SOURCE node.
struct MaskNode : arandu::ICpuScript<Bundle, Bundle> {
    void run(std::span<const Bundle> in, std::span<Bundle> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            Bundle b = in[i];
            const RawImage& r = *b.raw;
            auto cleaned = std::make_shared<std::vector<float>>(
                fnpost::binarize_mask_fast(r.segmentation.data(), r.h, r.w));
            b.segmentation_mask = std::make_shared<std::vector<uint8_t>>(
                fnpost::mask_up_u8(cleaned->data(), r.h, r.w));
            b.cleaned = cleaned;
            out[i] = std::move(b);
        }
    }
};

// B: segmentation -> quality (u8).
struct QualityNode : arandu::ICpuScript<Bundle, Bundle> {
    void run(std::span<const Bundle> in, std::span<Bundle> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            Bundle b = in[i];
            const RawImage& r = *b.raw;
            b.quality = std::make_shared<std::vector<uint8_t>>(
                fnpost::quality_u8(r.segmentation.data(), r.h, r.w));
            out[i] = std::move(b);
        }
    }
};

// C: orientation -> orientation_field (rad).
struct OriNode : arandu::ICpuScript<Bundle, Bundle> {
    void run(std::span<const Bundle> in, std::span<Bundle> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            Bundle b = in[i];
            const RawImage& r = *b.raw;
            b.orientation_field = std::make_shared<std::vector<float>>(
                fnpost::orientation_field(r.orientation.data(), 90, r.h, r.w));
            out[i] = std::move(b);
        }
    }
};

// D + F2: enhanced_real -> enhanced_image + enhanced_image_mod (needs cleaned).
struct EnhancedNode : arandu::ICpuScript<Bundle, Bundle> {
    void run(std::span<const Bundle> in, std::span<Bundle> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            Bundle b = in[i];
            const RawImage& r = *b.raw;
            b.enhanced_image = std::make_shared<std::vector<uint8_t>>(
                fnpost::enhanced_u8(r.enhanced_real.data(), r.H, r.W));
            auto up = fnpost::nearest_up(b.cleaned->data(), r.h, r.w);
            std::vector<float> mod(static_cast<size_t>(r.H) * r.W);
            for (size_t k = 0; k < mod.size(); ++k) mod[k] = r.enhanced_real[k] * up[k];
            b.enhanced_image_mod = std::make_shared<std::vector<uint8_t>>(
                fnpost::enhanced_u8(mod.data(), r.H, r.W));
            out[i] = std::move(b);
        }
    }
};

// E: minutiae (mask-modulated + unmod). SINK node (returns the full Bundle).
struct MinutiaeNode : arandu::ICpuScript<Bundle, Bundle> {
    void run(std::span<const Bundle> in, std::span<Bundle> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            Bundle b = in[i];
            const RawImage& r = *b.raw;
            int hw = r.h * r.w;
            std::vector<float> masked(hw);
            for (int k = 0; k < hw; ++k) masked[k] = r.minutiae_score[k] * (*b.cleaned)[k];
            b.minutiae = std::make_shared<std::vector<fnpost::Minutia>>(
                fnpost::detect_minutiae(masked.data(), r.minutiae_orientation.data(),
                                        r.minutiae_x_offset.data(), r.minutiae_y_offset.data(),
                                        r.h, r.w, r.threshold));
            b.minutiae_unmod = std::make_shared<std::vector<fnpost::Minutia>>(
                fnpost::detect_minutiae(r.minutiae_score.data(), r.minutiae_orientation.data(),
                                        r.minutiae_x_offset.data(), r.minutiae_y_offset.data(),
                                        r.h, r.w, r.threshold));
            out[i] = std::move(b);
        }
    }
};

// All postproc blocks fused into ONE node (best for the pipeline: fewer phases
// => fewer channels/threads + better cache locality). Same math as the chain.
struct PostprocNode : arandu::ICpuScript<Bundle, Bundle> {
    void run(std::span<const Bundle> in, std::span<Bundle> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            Bundle b = in[i];
            const RawImage& r = *b.raw;
            int hw = r.h * r.w, HW = r.H * r.W;
            auto cleaned = std::make_shared<std::vector<float>>(
                fnpost::binarize_mask_fast(r.segmentation.data(), r.h, r.w));
            b.segmentation_mask = std::make_shared<std::vector<uint8_t>>(fnpost::mask_up_u8(cleaned->data(), r.h, r.w));
            b.quality = std::make_shared<std::vector<uint8_t>>(fnpost::quality_u8(r.segmentation.data(), r.h, r.w));
            b.orientation_field = std::make_shared<std::vector<float>>(
                fnpost::orientation_field(r.orientation.data(), 90, r.h, r.w));
            b.enhanced_image = std::make_shared<std::vector<uint8_t>>(fnpost::enhanced_u8(r.enhanced_real.data(), r.H, r.W));
            auto up = fnpost::nearest_up(cleaned->data(), r.h, r.w);
            std::vector<float> mod(HW);
            for (int k = 0; k < HW; ++k) mod[k] = r.enhanced_real[k] * up[k];
            b.enhanced_image_mod = std::make_shared<std::vector<uint8_t>>(fnpost::enhanced_u8(mod.data(), r.H, r.W));
            std::vector<float> masked(hw);
            for (int k = 0; k < hw; ++k) masked[k] = r.minutiae_score[k] * (*cleaned)[k];
            b.minutiae = std::make_shared<std::vector<fnpost::Minutia>>(fnpost::detect_minutiae(
                masked.data(), r.minutiae_orientation.data(), r.minutiae_x_offset.data(), r.minutiae_y_offset.data(), r.h, r.w, r.threshold));
            b.minutiae_unmod = std::make_shared<std::vector<fnpost::Minutia>>(fnpost::detect_minutiae(
                r.minutiae_score.data(), r.minutiae_orientation.data(), r.minutiae_x_offset.data(), r.minutiae_y_offset.data(), r.h, r.w, r.threshold));
            b.cleaned = cleaned;
            out[i] = std::move(b);
        }
    }
};

// Build the postproc chain: mask(source) -> quality -> ori -> enhanced -> minutiae(sink).
inline arandu::Graph build_postproc_graph() {
    arandu::GraphBuilder gb;
    gb.add<Bundle, Bundle>("mask", std::make_shared<MaskNode>(), {});
    gb.add<Bundle, Bundle>("quality", std::make_shared<QualityNode>(), {"mask"});
    gb.add<Bundle, Bundle>("ori", std::make_shared<OriNode>(), {"quality"});
    gb.add<Bundle, Bundle>("enhanced", std::make_shared<EnhancedNode>(), {"ori"});
    gb.add<Bundle, Bundle>("minutiae", std::make_shared<MinutiaeNode>(), {"enhanced"});
    return gb.build();
}

}  // namespace fnaru
