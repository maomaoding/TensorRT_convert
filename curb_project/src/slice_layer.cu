#include "slice_layer.h"

__global__ void slicelayer(const int nthreads, const float* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, float* out_data) {
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

void SliceLayer::slicelayer_gpu(const int nthreads, const float* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, float* out_data)
    {
        slicelayer
        <<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>
        (nthreads, in_data, forward, num_slices, slice_size, bottom_slice_axis, top_slice_axis, offset_slice_axis, out_data);
    }