#include "pooling_layer.h"

__global__ void MaxPoolForward(const int nthreads,
	const float* const bottom_data, const int num, const int channels,
	const int height, const int width, const int pooled_height,
	const int pooled_width, const int kernel_h, const int kernel_w,
	const int stride_h, const int stride_w, const int pad_h, const int pad_w,
	float* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	const int pw = index % pooled_width;
	const int ph = (index / pooled_width) % pooled_height;
	const int c = (index / pooled_width / pooled_height) % channels;
	const int n = index / pooled_width / pooled_height / channels;
	int hstart = ph * stride_h - pad_h;
	int wstart = pw * stride_w - pad_w;
	const int hend = min(hstart + kernel_h, height);
	const int wend = min(wstart + kernel_w, width);
	hstart = max(hstart, 0);
	wstart = max(wstart, 0);
	float maxval = -FLT_MAX;
	int maxidx = -1;
	const float* const bottom_slice =
		bottom_data + (n * channels + c) * height * width;
	for (int h = hstart; h < hend; ++h) {
	  for (int w = wstart; w < wend; ++w) {
		if (bottom_slice[h * width + w] > maxval) {
		  maxidx = h * width + w;
		  maxval = bottom_slice[maxidx];
		}
	  }
	}
	top_data[index] = maxval;
  }
}

__global__ void AvePoolForward(const int nthreads,
	const float* const bottom_data, const int num, const int channels,
	const int height, const int width, const int pooled_height,
	const int pooled_width, const int kernel_h, const int kernel_w,
	const int stride_h, const int stride_w, const int pad_h, const int pad_w,
	float* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	const int pw = index % pooled_width;
	const int ph = (index / pooled_width) % pooled_height;
	const int c = (index / pooled_width / pooled_height) % channels;
	const int n = index / pooled_width / pooled_height / channels;
	int hstart = ph * stride_h - pad_h;
	int wstart = pw * stride_w - pad_w;
	int hend = min(hstart + kernel_h, height + pad_h);
	int wend = min(wstart + kernel_w, width + pad_w);
	const int pool_size = (hend - hstart) * (wend - wstart);
	hstart = max(hstart, 0);
	wstart = max(wstart, 0);
	hend = min(hend, height);
	wend = min(wend, width);
	float aveval = 0;
	const float* const bottom_slice =
		bottom_data + (n * channels + c) * height * width;
	for (int h = hstart; h < hend; ++h) {
	  for (int w = wstart; w < wend; ++w) {
		aveval += bottom_slice[h * width + w];
	  }
	}
	top_data[index] = aveval / pool_size;
  }
}

void PoolingLayer::MaxPoolForward_gpu(const int nthreads, const float* const bottom_data, const int num, const int channels,
								const int height, const int width, const int pooled_height,
								const int pooled_width, const int kernel_h, const int kernel_w,
								const int stride_h, const int stride_w, const int pad_h, const int pad_w,
								float* const top_data)
{
	MaxPoolForward<<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(nthreads, bottom_data, num, channels,
															height, width, pooled_height, pooled_width, kernel_h,
															kernel_w, stride_h, stride_w, pad_h, pad_w, top_data);
}

void PoolingLayer::AvePoolForward_gpu(const int nthreads, const float* const bottom_data, const int num, const int channels,
								const int height, const int width, const int pooled_height,
								const int pooled_width, const int kernel_h, const int kernel_w,
								const int stride_h, const int stride_w, const int pad_h, const int pad_w,
								float* const top_data)
{
	AvePoolForward<<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(nthreads, bottom_data, num, channels,
															height, width, pooled_height, pooled_width, kernel_h,
															kernel_w, stride_h, stride_w, pad_h, pad_w, top_data);
}