#include "layers/leaky_relu_layer.h"

template <typename Dtype>
__global__ void ReLUForward(int n, const Dtype *in, Dtype *out,
                            float negative_slope)
{

  CUDA_KERNEL_LOOP(index, n)
  {

    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}
void LeakyReluLayer::forward_gpu(int size, const float *inputs, float *outputs, float negative_slope)
{
  ReLUForward<float><<<TENSORRT_GET_BLOCKS(size), TENSORRT_CUDA_NUM_THREADS>>>(size,
                                                                               inputs, outputs, negative_slope);
}
