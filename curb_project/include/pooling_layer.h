#ifndef POOLING_LAYER_HPP_
#define POOLING_LAYER_HPP_

#include "utils.h"

enum PoolingParameter_RoundMode
{
	CEIL = 0,
	FLOOR = 1
};

enum PoolingParameter_PoolMethod
{
	MAX = 0,
	AVE = 1
};

class PoolingLayer : public nvinfer1::IPlugin
{
public:
	PoolingLayer(const int kernel_h_arg, const int kernel_w_arg,
				const int stride_h_, const int stride_w_,
				const int pad_h_, const int pad_w_,
				const bool global_pooling_, PoolingParameter_RoundMode round_mode_, PoolingParameter_PoolMethod pool_method_);
	PoolingLayer(const void* buffer, size_t size);
	inline int getNbOutputs() const override { return 1; };
	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;
	int initialize() override{ return 0; };
	inline void terminate() override{}
	inline size_t getWorkspaceSize(int) const override { return 0; }
	int enqueue(int batchSize, const void* const *inputs, void** outputs, void*, cudaStream_t stream) override;
	size_t getSerializationSize() override;
	void serialize(void* buffer) override;
	void configure(const nvinfer1::Dims*inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOuputs, int) override;
	void MaxPoolForward_gpu(const int nthreads, const float* const bottom_data, const int num, const int channels,
							const int height, const int width, const int pooled_height,
							const int pooled_width, const int kernel_h, const int kernel_w,
							const int stride_h, const int stride_w, const int pad_h, const int pad_w,
							float* const top_data);
	void AvePoolForward_gpu(const int nthreads, const float* const bottom_data, const int num, const int channels,
							const int height, const int width, const int pooled_height,
							const int pooled_width, const int kernel_h, const int kernel_w,
							const int stride_h, const int stride_w, const int pad_h, const int pad_w,
							float* const top_data);

private:
	bool global_pooling_;
	int kernel_h_, kernel_w_;
	int stride_h_, stride_w_;
	int pad_h_, pad_w_;
	int pooled_height_, pooled_width_;
	int channels_;
	int height_, width_;
	PoolingParameter_RoundMode round_mode_;
	PoolingParameter_PoolMethod pool_method_;
};

#endif