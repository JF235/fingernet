// fingernet-on-arandu umbrella: one include for a driver that builds the extraction
// graph. Brings the whole arandu contract (graph, both executors, policy, loader,
// profiler) plus every fingernet node that goes in it:
//
//   PathItem --load--> InputImage --onnx--> Bundle --postproc--> Bundle --serialize--> Written
//
// Needs the arandu headers on the include path, and libpng for the I/O nodes. The
// ONNX node self-gates on FINGERNET_WITH_ONNX, so including this without ONNX
// Runtime is fine -- FingernetOnnx simply is not declared, and a driver that names
// it fails to compile, which is the honest outcome.
//
// For the kernels alone -- no graph, no arandu, no libpng -- include fingernet.hpp.
#pragma once
#include "arandu/arandu.hpp"

#include "arandu_nodes.hpp"
#include "fingernet.hpp"
#include "io_nodes.hpp"
#include "onnx_model.hpp"
