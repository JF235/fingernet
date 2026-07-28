// .min serialization: converts raw FingerNet minutiae (x, y, angle_rad,
// score in [0,1]) to the standard .min convention (integer CCW degrees in
// [0,360), integer quality 0..100) and writes the file. Mirrors the exact
// conversion in pytorch/fingernet/api.py:save_results.
#pragma once
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "postproc.hpp"

namespace fnmin {

struct MinRow { long x, y, angle, quality; };

// numpy-style real modulo: result has the sign of the divisor -> [0, m).
inline double pymod(double a, double m) {
    double r = std::fmod(a, m);
    if (r < 0) r += m;
    return r;
}

// api.py: angle = round( (-rad2deg(a)) % 360 ) ; quality = round(score*100)
//         x, y   = int(x), int(y)   (truncation)
// np.round is banker's rounding -> std::nearbyint (default nearest-even).
inline MinRow to_min(const fnpost::Minutia& m) {
    double deg = -(static_cast<double>(m.angle) * 180.0 / fnpost::PI);
    MinRow r;
    r.angle = static_cast<long>(std::nearbyint(pymod(deg, 360.0)));
    r.quality = static_cast<long>(std::nearbyint(static_cast<double>(m.score) * 100.0));
    r.x = static_cast<long>(std::trunc(m.x));  // int() truncation toward zero
    r.y = static_cast<long>(std::trunc(m.y));
    return r;
}

inline std::vector<MinRow> to_min_rows(const std::vector<fnpost::Minutia>& mnt) {
    std::vector<MinRow> rows;
    rows.reserve(mnt.size());
    for (const auto& m : mnt) rows.push_back(to_min(m));
    return rows;
}

// np.savetxt(..., fmt="%d", header="X Y ANGLE QUALITY", comments="#MIN ")
// -> first line "#MIN X Y ANGLE QUALITY", then space-separated integer rows.
inline void write_min(const std::string& path, const std::vector<MinRow>& rows) {
    std::ofstream f(path);
    f << "#MIN X Y ANGLE QUALITY\n";
    for (const auto& r : rows)
        f << r.x << " " << r.y << " " << r.angle << " " << r.quality << "\n";
}

}  // namespace fnmin
