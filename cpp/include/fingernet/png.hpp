// Minimal 8-bit grayscale PNG writer (libpng). PNG is lossless, so any correct
// encoder produces a file that decodes back to the exact pixel array -> the
// parity gate is on DECODED pixels, never on file bytes (PIL/zlib settings
// differ from libpng and would never byte-match, without any pixel difference).
#pragma once
#include <png.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace fnpng {

// Read an 8-bit grayscale PNG (expands palette/RGB/<8bpp to 8-bit gray, strips
// alpha/16-bit). SD258 is already 8-bit gray so this is a direct read matching
// PIL's Image.open(...).convert("L").
inline std::vector<uint8_t> read_gray(const std::string& path, int& H, int& W) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("png: cannot open " + path);
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (!png || !info || setjmp(png_jmpbuf(png))) {
        if (png) png_destroy_read_struct(&png, info ? &info : nullptr, nullptr);
        std::fclose(fp);
        throw std::runtime_error("png: read failure for " + path);
    }
    png_init_io(png, fp);
    png_read_info(png, info);
    W = static_cast<int>(png_get_image_width(png, info));
    H = static_cast<int>(png_get_image_height(png, info));
    png_byte color = png_get_color_type(png, info);
    png_byte depth = png_get_bit_depth(png, info);
    if (depth == 16) png_set_strip_16(png);
    if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (color & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(png);
    if (color == PNG_COLOR_TYPE_RGB || color == PNG_COLOR_TYPE_RGB_ALPHA)
        png_set_rgb_to_gray_fixed(png, 1, -1, -1);  // ITU-R 601-2 luma (PIL "L")
    png_read_update_info(png, info);
    std::vector<uint8_t> img(static_cast<size_t>(H) * W);
    std::vector<png_bytep> rows(static_cast<size_t>(H));
    for (int y = 0; y < H; ++y) rows[y] = img.data() + static_cast<size_t>(y) * W;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);
    return img;
}

inline void write_gray(const std::string& path, const uint8_t* data, int H, int W) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("png: cannot open " + path);
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (!png || !info || setjmp(png_jmpbuf(png))) {
        if (png) png_destroy_write_struct(&png, info ? &info : nullptr);
        std::fclose(fp);
        throw std::runtime_error("png: write failure for " + path);
    }
    png_init_io(png, fp);
    png_set_IHDR(png, info, static_cast<png_uint_32>(W), static_cast<png_uint_32>(H), 8,
                 PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    std::vector<png_bytep> rows(static_cast<size_t>(H));
    for (int y = 0; y < H; ++y)
        rows[y] = const_cast<png_bytep>(data + static_cast<size_t>(y) * W);
    png_write_image(png, rows.data());
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    std::fclose(fp);
}

}  // namespace fnpng
