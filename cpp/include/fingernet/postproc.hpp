// FingerNet post-processing, ported to C++ (CPU), item-pure (one image at a
// time). Mirrors pytorch/fingernet/wrapper.py exactly; see docs for the
// parity-critical details (banker's rounding, first-max argmax, elementwise
// NMS distance, bilinear align_corners=False, .byte() truncation).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace fnpost {

constexpr double PI = 3.14159265358979323846;

// ---- helpers ---------------------------------------------------------------

// torch.round == round-half-to-even. std::nearbyint honours the default
// FE_TONEAREST rounding mode (nearest-even) -> exact match.
inline float round_even(float v) { return std::nearbyint(v); }

// .byte(): cast truncates toward zero. Input already in [0,255].
inline uint8_t to_u8_trunc(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 255.0f) return 255;
    return static_cast<uint8_t>(v);  // truncation toward zero
}

// argmax over the channel dim, per cell; returns the FIRST maximal index.
//
// The pipeline does NOT call this: the decode's four argmaxes run inside the ONNX
// graph (ONNX ArgMax defaults to select_last_index=0, the same first-maximal rule).
// It stays here because the parity harness reads the Python dump of the float channel
// stacks, which predates that change, and has to reduce them itself to compare like
// with like.
inline std::vector<int32_t> argmax_plane(const float* t, int C, int h, int w) {
    const int hw = h * w;
    std::vector<int32_t> idx(static_cast<size_t>(hw));
    for (int i = 0; i < hw; ++i) {
        int best = 0;
        float bv = t[i];  // channel 0
        for (int c = 1; c < C; ++c) {
            float v = t[static_cast<size_t>(c) * hw + i];
            if (v > bv) { bv = v; best = c; }
        }
        idx[i] = best;
    }
    return idx;
}

// Bin -> radians, for both angle heads: the orientation field's 90 bins and the
// minutia's 180. Both are 2 degrees wide with -89 at bin 0's centre.
inline float bin_to_angle(int bin) {
    return static_cast<float>((bin * 2.0 - 89.0) * PI / 180.0);
}

// ---- Block A: binary segmentation mask ------------------------------------
// _post_binarize_mask_fast: round -> separable gaussian(5,1.5) -> round.
inline std::vector<float> gaussian_kernel_5() {
    // matches _get_gaussian_kernel1d(5, 1.5) in float32
    std::vector<float> g(5);
    float sigma = 1.5f, sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        float c = static_cast<float>(i) - 2.0f;   // coords - kernel_size//2
        g[i] = std::exp(-(c * c) / (2.0f * sigma * sigma));
        sum += g[i];
    }
    for (auto& v : g) v /= sum;
    return g;
}

// seg: [h,w]. out cleaned: [h,w] in {0,1}.
inline std::vector<float> binarize_mask_fast(const float* seg, int h, int w) {
    static const std::vector<float> ker = gaussian_kernel_5();
    std::vector<float> bin(static_cast<size_t>(h) * w);
    for (size_t i = 0; i < bin.size(); ++i) bin[i] = round_even(seg[i]);  // {0,1}

    // separable zero-padded conv, radius 2: horizontal then vertical.
    std::vector<float> bh(static_cast<size_t>(h) * w, 0.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int j = 0; j < 5; ++j) {
                int xx = x + j - 2;
                if (xx >= 0 && xx < w) acc += ker[j] * bin[static_cast<size_t>(y) * w + xx];
            }
            bh[static_cast<size_t>(y) * w + x] = acc;
        }
    std::vector<float> cleaned(static_cast<size_t>(h) * w, 0.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int j = 0; j < 5; ++j) {
                int yy = y + j - 2;
                if (yy >= 0 && yy < h) acc += ker[j] * bh[static_cast<size_t>(yy) * w + x];
            }
            cleaned[static_cast<size_t>(y) * w + x] = round_even(acc);
        }
    return cleaned;
}

// cleaned [h,w] -> uint8 [H,W] (nearest x8, *255).
inline std::vector<uint8_t> mask_up_u8(const float* cleaned, int h, int w) {
    int H = h * 8, W = w * 8;
    std::vector<uint8_t> out(static_cast<size_t>(H) * W);
    for (int Y = 0; Y < H; ++Y)
        for (int X = 0; X < W; ++X)
            out[static_cast<size_t>(Y) * W + X] =
                to_u8_trunc(cleaned[static_cast<size_t>(Y / 8) * w + (X / 8)] * 255.0f);
    return out;
}

// cleaned coarse -> float [H,W] nearest-up (reused by _mod outputs)
inline std::vector<float> nearest_up(const float* src, int h, int w) {
    int H = h * 8, W = w * 8;
    std::vector<float> out(static_cast<size_t>(H) * W);
    for (int Y = 0; Y < H; ++Y)
        for (int X = 0; X < W; ++X)
            out[static_cast<size_t>(Y) * W + X] = src[static_cast<size_t>(Y / 8) * w + (X / 8)];
    return out;
}

// ---- Block B: continuous quality mask -------------------------------------
// interpolate(seg, x8, bilinear, align_corners=False) * 255 -> u8.
// Two-stage separable form matching aten's upsample_bilinear2d (h then w
// lambdas). Residual: a handful of pixels per multi-megapixel map land on the
// *255 truncation boundary and flip by +-1 vs torch (float accumulation order);
// this is below FingerNet's own run-to-run reproducibility on the quality map.
inline std::vector<uint8_t> quality_u8(const float* seg, int h, int w) {
    int H = h * 8, W = w * 8;
    std::vector<uint8_t> out(static_cast<size_t>(H) * W);
    const float scale = 1.0f / 8.0f;
    for (int Y = 0; Y < H; ++Y) {
        float sy = (static_cast<float>(Y) + 0.5f) * scale - 0.5f;
        if (sy < 0) sy = 0;
        int y0 = static_cast<int>(std::floor(sy));
        float wy = sy - y0;
        int y1 = std::min(y0 + 1, h - 1);
        for (int X = 0; X < W; ++X) {
            float sx = (static_cast<float>(X) + 0.5f) * scale - 0.5f;
            if (sx < 0) sx = 0;
            int x0 = static_cast<int>(std::floor(sx));
            float wx = sx - x0;
            int x1 = std::min(x0 + 1, w - 1);
            float p00 = seg[static_cast<size_t>(y0) * w + x0];
            float p01 = seg[static_cast<size_t>(y0) * w + x1];
            float p10 = seg[static_cast<size_t>(y1) * w + x0];
            float p11 = seg[static_cast<size_t>(y1) * w + x1];
            float top = p00 * (1 - wx) + p01 * wx;
            float bot = p10 * (1 - wx) + p11 * wx;
            float v = top * (1 - wy) + bot * wy;
            out[static_cast<size_t>(Y) * W + X] = to_u8_trunc(v * 255.0f);
        }
    }
    return out;
}

// ---- Block C: orientation field -------------------------------------------
// ori_idx[h,w] (argmax bins, from the graph) -> nearest-up x8 -> radians [H,W]
inline std::vector<float> orientation_field(const int32_t* ori_idx, int h, int w) {
    // Decode once per coarse cell, then let nearest_up own the x8 rule (as every
    // other coarse->full product does): 64x fewer bin_to_angle calls than decoding
    // per pixel, at the cost of one [h*w] temporary.
    std::vector<float> coarse(static_cast<size_t>(h) * w);
    for (size_t i = 0; i < coarse.size(); ++i) coarse[i] = bin_to_angle(ori_idx[i]);
    return nearest_up(coarse.data(), h, w);
}

// ---- Block D: enhanced image ----------------------------------------------
// _normalize_minmax_to_uint8: per-image min-max -> *255 -> u8
inline std::vector<uint8_t> enhanced_u8(const float* enh, int H, int W) {
    size_t n = static_cast<size_t>(H) * W;
    float mn = enh[0], mx = enh[0];
    for (size_t i = 1; i < n; ++i) { mn = std::min(mn, enh[i]); mx = std::max(mx, enh[i]); }
    float denom = (mx - mn) + 1e-8f;
    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = to_u8_trunc(((enh[i] - mn) / denom) * 255.0f);
    return out;
}

// The masked variant, normalised AFTER masking. It is a separate product and not
// `enhanced_u8` times the mask: min-max over the masked float maps the zeroed
// background to mid-grey (the Gabor minimum is negative), where masking the
// quantised u8 would put it at black. Fused so neither the upsampled mask nor the
// masked float is ever materialised -- two [H,W] float temporaries saved per image.
inline std::vector<uint8_t> enhanced_masked_u8(const float* enh, const float* cleaned,
                                               int h, int w) {
    int H = h * 8, W = w * 8;
    auto at = [&](int Y, int X) {
        return enh[static_cast<size_t>(Y) * W + X] * cleaned[static_cast<size_t>(Y / 8) * w + (X / 8)];
    };
    float mn = at(0, 0), mx = mn;
    for (int Y = 0; Y < H; ++Y)
        for (int X = 0; X < W; ++X) { float v = at(Y, X); mn = std::min(mn, v); mx = std::max(mx, v); }
    float denom = (mx - mn) + 1e-8f;
    std::vector<uint8_t> out(static_cast<size_t>(H) * W);
    for (int Y = 0; Y < H; ++Y)
        for (int X = 0; X < W; ++X)
            out[static_cast<size_t>(Y) * W + X] = to_u8_trunc(((at(Y, X) - mn) / denom) * 255.0f);
    return out;
}

// ---- Block E: minutiae detection + NMS ------------------------------------
struct Minutia { float x, y, angle, score; };

// score[h,w] already mask-modulated. ori_idx/xoff_idx/yoff_idx are the [h,w] argmax
// planes the model returns. Returns NMS-kept minutiae in score-desc order.
inline std::vector<Minutia> detect_minutiae(
    const float* score, const int32_t* ori_idx, const int32_t* xoff_idx,
    const int32_t* yoff_idx, int h, int w, float threshold,
    float dist_thresh = 16.0f, float angle_thresh = static_cast<float>(PI / 6.0)) {
    std::vector<Minutia> cand;
    // row-major candidate order == torch.where(score > threshold)
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            float s = score[i];
            if (s > threshold) {
                Minutia m;
                m.x = static_cast<float>(x) * 8.0f + static_cast<float>(xoff_idx[i]);
                m.y = static_cast<float>(y) * 8.0f + static_cast<float>(yoff_idx[i]);
                m.angle = bin_to_angle(ori_idx[i]);
                m.score = s;
                cand.push_back(m);
            }
        }
    if (cand.empty()) return cand;

    // sort by score desc; stable tie-break by original index (see parity note)
    std::vector<int> order(cand.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return cand[a].score > cand[b].score; });
    std::vector<Minutia> s(cand.size());
    for (size_t k = 0; k < order.size(); ++k) s[k] = cand[order[k]];

    // greedy NMS: elementwise distance (NOT the cdist factored form)
    int n = static_cast<int>(s.size());
    std::vector<char> keep(n, 1);
    for (int i = 0; i < n; ++i) {
        if (!keep[i]) continue;
        for (int j = i + 1; j < n; ++j) {
            if (!keep[j]) continue;
            float dx = s[i].x - s[j].x, dy = s[i].y - s[j].y;
            float dist = std::sqrt(dx * dx + dy * dy);
            float ad = std::fabs(s[i].angle - s[j].angle);
            float ang = std::min(ad, static_cast<float>(2.0 * PI) - ad);
            if (dist < dist_thresh && ang < angle_thresh) keep[j] = 0;
        }
    }
    std::vector<Minutia> out;
    for (int i = 0; i < n; ++i)
        if (keep[i]) out.push_back(s[i]);
    return out;
}

}  // namespace fnpost
