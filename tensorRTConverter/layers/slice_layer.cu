#include "layers/slice_layer.h"
#include <iostream>


template <typename Dtype>
__global__ void Slice(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_slice_size = slice_size * top_slice_axis;
    const int slice_num = index / total_slice_size;
    const int slice_index = index % total_slice_size;
    const int bottom_index = slice_index +
        (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
    if (forward) {
      out_data[index] = in_data[bottom_index];
    } else {
      out_data[bottom_index] = in_data[index];
    }
  }
}

void SliceLayer::forward_gpu(const float* in_data,void** out_data,const int num_slices,const int slice_size,int slice_axis,vector<int> bottom_shape,vector<int> output_shape,int output_shape_size) {
  int offset_slice_axis = 0;
  const int bottom_slice_axis = bottom_shape[slice_axis];
  bool kForward=true;

  for (int i = 0; i < output_shape_size; ++i) {
    float* output=reinterpret_cast<float*>(out_data[i]);
    const int top_slice_axis =output_shape[i];
    const int top_slice_size = top_slice_axis * slice_size;
    const int nthreads = top_slice_size * num_slices;

    Slice<float>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<TENSORRT_GET_BLOCKS(nthreads), TENSORRT_CUDA_NUM_THREADS>>>(
        nthreads, in_data, kForward, num_slices, slice_size,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, output);
    offset_slice_axis += top_slice_axis;
  }
  return ;
}
