#include "layers/upsample_layer.h"

__device__ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor)
{
    int x, y, z, w;
    w = ii % d3;
    ii = ii / d3;
    z = ii % d2;
    ii = ii / d2;
    y = ii % d1;
    ii = ii / d1;
    x = ii;
    w = w / scale_factor;
    z = z / scale_factor;
    d2 /= scale_factor;
    d3 /= scale_factor;
    return (((x * d1 + y) * d2) + z) * d3 + w;
}

__global__ void upscale(const float *input, float *output,
                        int no_elements, int scale_factor, int d1, int d2, int d3)
{
    int ii = threadIdx.x + blockDim.x * blockIdx.x;
    if (ii >= no_elements)
        return;
    int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);
    output[ii] = input[ipidx];
}

void UpsampleLayer::forward_gpu(const float *input, float *output, float scale, int N, int C, int H, int W)
{
    int numElem = N * C * H * W;
    upscale<<<TENSORRT_GET_BLOCKS(numElem), TENSORRT_CUDA_NUM_THREADS>>>(input, output, numElem, scale, C, H, W);
}
