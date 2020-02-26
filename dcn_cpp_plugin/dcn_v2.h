
#ifndef DCN_V2_H
#define DCN_V2_H
#include <torch/extension.h>

#ifdef __cplusplus
extern "C"
{
#endif
    

    at::Tensor
    dcn_v2_cuda_forward(const at::Tensor &input,
                        const at::Tensor &weight,
                        const at::Tensor &bias,
                        const at::Tensor &offset,
                        const at::Tensor &mask,
                        const int64_t kernel_h,
                        const int64_t kernel_w,
                        const int64_t stride_h,
                        const int64_t stride_w,
                        const int64_t pad_h,
                        const int64_t pad_w,
                        const int64_t dilation_h,
                        const int64_t dilation_w,
                        const int64_t deformable_group);

#ifdef __cplusplus
}
#endif

#endif