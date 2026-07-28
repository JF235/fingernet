// Minimal .npy reader (v1/v2, C-order, little-endian). Enough for the
// postproc parity harness: float32 ('<f4'), uint8 ('|u1'), int64 ('<i8').
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fnpy {

struct Array {
    std::vector<size_t> shape;
    std::string descr;      // e.g. "<f4", "|u1", "<i8"
    std::vector<char> raw;  // raw little-endian bytes

    size_t size() const {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        return n;
    }
    template <typename T>
    const T* as() const { return reinterpret_cast<const T*>(raw.data()); }
};

inline Array load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy: bad magic in " + path);
    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t h = 0;
        f.read(reinterpret_cast<char*>(&h), 2);
        hlen = h;
    } else {
        uint32_t h = 0;
        f.read(reinterpret_cast<char*>(&h), 4);
        hlen = h;
    }
    std::string header(hlen, '\0');
    f.read(&header[0], hlen);

    Array a;
    // descr
    auto dp = header.find("'descr'");
    auto q1 = header.find('\'', header.find(':', dp) + 1);
    auto q2 = header.find('\'', q1 + 1);
    a.descr = header.substr(q1 + 1, q2 - q1 - 1);
    // shape
    auto sp = header.find("'shape'");
    auto lp = header.find('(', sp);
    auto rp = header.find(')', lp);
    std::string tup = header.substr(lp + 1, rp - lp - 1);
    size_t pos = 0;
    while (pos < tup.size()) {
        while (pos < tup.size() && (tup[pos] == ' ' || tup[pos] == ',')) ++pos;
        if (pos >= tup.size()) break;
        size_t start = pos;
        while (pos < tup.size() && tup[pos] >= '0' && tup[pos] <= '9') ++pos;
        if (pos > start) a.shape.push_back(std::stoul(tup.substr(start, pos - start)));
    }
    // itemsize from descr (last char is byte count)
    size_t itemsize = static_cast<size_t>(a.descr.back() - '0');
    a.raw.resize(a.size() * itemsize);
    f.read(a.raw.data(), static_cast<std::streamsize>(a.raw.size()));
    if (!f) throw std::runtime_error("npy: short read in " + path);
    return a;
}

}  // namespace fnpy
