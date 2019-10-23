
#include "dcn_v2.h"
#include <torch/script.h>

// static auto registry =
//     torch::jit::RegisterOperators("my_ops::dcn_v2_forward", &dcn_v2_forward)
//         .op("my_ops::dcn_v2_backward", &dcn_v2_backward)
//         .op("my_ops::dcn_v2_psroi_pooling_forward", &dcn_v2_psroi_pooling_forward)
//         .op("my_ops::dcn_v2_psroi_pooling_backward", &dcn_v2_psroi_pooling_backward);

static auto registry =
    torch::jit::RegisterOperators("my_ops::dcn_v2_cuda_forward_v2", &dcn_v2_cuda_forward_v2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("dcn_v2_forward", &dcn_v2_forward, "dcn_v2_forward");
  m.def("dcn_v2_backward", &dcn_v2_backward, "dcn_v2_backward");
  m.def("dcn_v2_psroi_pooling_forward", &dcn_v2_psroi_pooling_forward, "dcn_v2_psroi_pooling_forward");
  m.def("dcn_v2_psroi_pooling_backward", &dcn_v2_psroi_pooling_backward, "dcn_v2_psroi_pooling_backward");
}
