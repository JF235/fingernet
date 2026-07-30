// FingerNet postproc blocks wrapped as arandu nodes (ICpuScript<Bundle,Bundle>),
// item-pure (out[i] depends only on in[i]). The blocks are chained; each node
// adds its output to a shared-ptr Bundle carrier so chaining copies stay cheap.
// This is the plug-and-play adapter: arandu builds the graph, fingernet owns
// the kernels. Requires the arandu headers on the include path.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "arandu/graph.hpp"
#include "arandu/kernel.hpp"
#include "postproc.hpp"

namespace fnaru {

// The model's outputs for ONE image (single-channel dims squeezed away).
//
// Four of the seven are INDEX planes, not the float channel stacks the network
// produces: orientation (90 bins), minutiae orientation (180) and the two 8-bin
// offsets exist only to be argmax'd, so the argmax happens in the ONNX graph and
// the host receives the winning bin. That is 286 coarse channels not transferred,
// 4.4 MB per 512x512 image, and it is why RawImage is 1.15 MB instead of 5.8.
struct RawImage {
    int h = 0, w = 0, H = 0, W = 0;
    int orig_h = 0, orig_w = 0;   // pre-pad dims (output cropping)
    std::string id;               // relative id for output filenames
    float threshold = 0.05f;
    std::vector<float> segmentation;              // [h*w] continuous sigmoid
    std::vector<float> enhanced_real;             // [H*W]
    std::vector<float> minutiae_score;            // [h*w]
    std::vector<int32_t> orientation_index;       // [h*w] winning bin of 90
    std::vector<int32_t> minutiae_orientation_index;  // [h*w] winning bin of 180
    std::vector<int32_t> minutiae_x_index;        // [h*w] winning bin of 8
    std::vector<int32_t> minutiae_y_index;        // [h*w] winning bin of 8
};

// A preprocessed input image for the model: [H*W] grayscale/255, padded to /8.
struct InputImage {
    int h = 0, w = 0, H = 0, W = 0;   // coarse (H/8) + full dims after padding
    int orig_h = 0, orig_w = 0;       // pre-pad dims (for output cropping)
    std::string id;                   // relative id for output filenames
    std::vector<float> data;          // [H*W]
};

// Carrier threaded through the postproc chain; shared_ptr => cheap to copy.
// The five primitives are what wrapper.py returns; enhanced_image_mod is the one
// derived product that is NOT the mask times a primitive (normalising the masked
// float puts the background at mid-grey, masking the quantised u8 puts it at
// black), so it is computed here or not at all -- see the `full` flag.
struct Bundle {
    std::shared_ptr<const RawImage> raw;
    std::shared_ptr<const std::vector<float>> cleaned;               // coarse {0,1}
    std::shared_ptr<const std::vector<uint8_t>> segmentation_mask;   // [H*W]
    std::shared_ptr<const std::vector<uint8_t>> quality;             // [H*W]
    std::shared_ptr<const std::vector<float>> orientation_field;     // [H*W] rad
    std::shared_ptr<const std::vector<uint8_t>> enhanced_image;      // [H*W]
    std::shared_ptr<const std::vector<uint8_t>> enhanced_image_mod;  // [H*W], full only
    std::shared_ptr<const std::vector<fnpost::Minutia>> minutiae;
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

// C: orientation_index -> orientation_field (rad).
struct OriNode : arandu::ICpuScript<Bundle, Bundle> {
    void run(std::span<const Bundle> in, std::span<Bundle> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            Bundle b = in[i];
            const RawImage& r = *b.raw;
            b.orientation_field = std::make_shared<std::vector<float>>(
                fnpost::orientation_field(r.orientation_index.data(), r.h, r.w));
            out[i] = std::move(b);
        }
    }
};

// D (+ the mod variant under `full`): enhanced_real -> enhanced_image. Needs cleaned.
struct EnhancedNode : arandu::ICpuScript<Bundle, Bundle> {
    bool full = false;
    void run(std::span<const Bundle> in, std::span<Bundle> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            Bundle b = in[i];
            const RawImage& r = *b.raw;
            b.enhanced_image = std::make_shared<std::vector<uint8_t>>(
                fnpost::enhanced_u8(r.enhanced_real.data(), r.H, r.W));
            if (full) b.enhanced_image_mod = std::make_shared<std::vector<uint8_t>>(
                fnpost::enhanced_masked_u8(r.enhanced_real.data(), b.cleaned->data(), r.h, r.w));
            out[i] = std::move(b);
        }
    }
};

// E: minutiae. SINK node (returns the full Bundle).
struct MinutiaeNode : arandu::ICpuScript<Bundle, Bundle> {
    void run(std::span<const Bundle> in, std::span<Bundle> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            Bundle b = in[i];
            b.minutiae = std::make_shared<std::vector<fnpost::Minutia>>(
                detect(*b.raw, *b.cleaned));
            out[i] = std::move(b);
        }
    }
    // Gating the score with the mask BEFORE the NMS is load-bearing, not a shortcut:
    // the NMS is order-dependent, so a background candidate that survives into it can
    // suppress a real foreground minutia. Masking afterwards is a different operation.
    static std::vector<fnpost::Minutia> detect(const RawImage& r, const std::vector<float>& cleaned) {
        int hw = r.h * r.w;
        std::vector<float> masked(hw);
        for (int k = 0; k < hw; ++k) masked[k] = r.minutiae_score[k] * cleaned[k];
        return fnpost::detect_minutiae(masked.data(), r.minutiae_orientation_index.data(),
                                       r.minutiae_x_index.data(), r.minutiae_y_index.data(),
                                       r.h, r.w, r.threshold);
    }
};

// All postproc blocks fused into ONE node (best for the pipeline: fewer phases
// => fewer channels/threads + better cache locality). Same math as the chain.
struct PostprocNode : arandu::ICpuScript<Bundle, Bundle> {
    bool full = false;
    void run(std::span<const Bundle> in, std::span<Bundle> out, arandu::RunCtx&) const override {
        for (size_t i = 0; i < in.size(); ++i) {
            Bundle b = in[i];
            const RawImage& r = *b.raw;
            auto cleaned = std::make_shared<std::vector<float>>(
                fnpost::binarize_mask_fast(r.segmentation.data(), r.h, r.w));
            b.segmentation_mask = std::make_shared<std::vector<uint8_t>>(fnpost::mask_up_u8(cleaned->data(), r.h, r.w));
            b.quality = std::make_shared<std::vector<uint8_t>>(fnpost::quality_u8(r.segmentation.data(), r.h, r.w));
            b.orientation_field = std::make_shared<std::vector<float>>(
                fnpost::orientation_field(r.orientation_index.data(), r.h, r.w));
            b.enhanced_image = std::make_shared<std::vector<uint8_t>>(fnpost::enhanced_u8(r.enhanced_real.data(), r.H, r.W));
            if (full) b.enhanced_image_mod = std::make_shared<std::vector<uint8_t>>(
                fnpost::enhanced_masked_u8(r.enhanced_real.data(), cleaned->data(), r.h, r.w));
            b.minutiae = std::make_shared<std::vector<fnpost::Minutia>>(MinutiaeNode::detect(r, *cleaned));
            b.cleaned = cleaned;
            out[i] = std::move(b);
        }
    }
};

// Build the postproc chain: mask(source) -> quality -> ori -> enhanced -> minutiae(sink).
inline arandu::Graph build_postproc_graph(bool full = false) {
    auto enh = std::make_shared<EnhancedNode>(); enh->full = full;
    arandu::GraphBuilder gb;
    gb.add<Bundle, Bundle>("mask", std::make_shared<MaskNode>(), {});
    gb.add<Bundle, Bundle>("quality", std::make_shared<QualityNode>(), {"mask"});
    gb.add<Bundle, Bundle>("ori", std::make_shared<OriNode>(), {"quality"});
    gb.add<Bundle, Bundle>("enhanced", enh, {"ori"});
    gb.add<Bundle, Bundle>("minutiae", std::make_shared<MinutiaeNode>(), {"enhanced"});
    return gb.build();
}

}  // namespace fnaru
