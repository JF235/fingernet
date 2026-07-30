// fingernet umbrella: the post-processing kernels and the .min serialization --
// everything that needs nothing but a C++20 compiler. Include this to decode a
// FingerNet forward pass into products, with no opinion about who ran the model or
// how the work is scheduled.
//
// Deliberately NOT here, because each needs a build option and would break a plain
// `#include <fingernet/fingernet.hpp>` for everyone else:
//   pipeline.hpp     the arandu headers (and, for its ONNX node, ONNX Runtime) --
//                    that is the OTHER umbrella, for a driver that builds a graph
//   png.hpp          libpng
//   npy.hpp          nothing, but it exists only to read the Python reference dump
#pragma once
#include "minfmt.hpp"
#include "postproc.hpp"
