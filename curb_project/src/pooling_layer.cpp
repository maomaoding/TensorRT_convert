#include "pooling_layer.h"

PoolingLayer::PoolingLayer(const int kernel_h_arg, const int kernel_w_arg,
						const int stride_h_, const int stride_w_,
						const int pad_h_, const int pad_w_,
						const bool global_pooling_, PoolingParameter_RoundMode round_mode_,
						PoolingParameter_PoolMethod pool_method_):
						kernel_h_(kernel_h_arg), kernel_w_(kernel_w_arg),
						stride_h_(stride_h_), stride_w_(stride_w_),
						pad_h_(pad_h_), pad_w_(pad_w_),
						global_pooling_(global_pooling_), round_mode_(round_mode_), pool_method_(pool_method_),
						channels_(64), height_(240), width_(240)
{
	if(global_pooling_)
	{
		kernel_w_ = width_;
		kernel_h_ = height_;
	}else{
		switch(round_mode_){
		case CEIL:
			pooled_height_ = static_cast<int>(ceil(static_cast<float>(
				height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
			pooled_width_ = static_cast<int>(ceil(static_cast<float>(
				width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
			break;
		case FLOOR:
			pooled_height_ = static_cast<int>(floor(static_cast<float>(
				height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
			pooled_width_ = static_cast<int>(floor(static_cast<float>(
				width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
			break;
		}
		if(pad_h_ || pad_w_)
		{
			if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_){
				--pooled_height_;
			}
			if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_){
				--pooled_width_;
			}
		}
	}
}

PoolingLayer::PoolingLayer(const void* buffer, size_t size)
{
	const char* d = reinterpret_cast<const char*>(buffer);
	global_pooling_ = d[0];
	const int* dd = reinterpret_cast<const int*>(d);
	kernel_h_ = dd[1];
	kernel_w_ = dd[2];
	stride_h_ = dd[3];
	stride_w_ = dd[4];
	pad_h_ = dd[5];
	pad_w_ = dd[6];
	pooled_height_ = dd[7];
	pooled_width_ = dd[8];
	channels_ = dd[9];
	height_ = dd[10];
	width_ = dd[11];
	round_mode_ = (PoolingParameter_RoundMode)dd[12];
	pool_method_ = (PoolingParameter_PoolMethod)dd[13];
}

nvinfer1::Dims PoolingLayer::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
	return nvinfer1::DimsCHW(channels_, pooled_height_, pooled_width_);
}

int PoolingLayer::enqueue(int batchSize, const void* const *inputs, void** outputs, void*, cudaStream_t stream)
{
	const float* bottom_data = reinterpret_cast<const float*>(inputs[0]);
	float* top_data = reinterpret_cast<float*>(outputs[0]);
	int count = 1*channels_*pooled_height_*pooled_width_;
	switch (pool_method_){
	case MAX:
		PoolingLayer::MaxPoolForward_gpu(
			count, bottom_data, batchSize, channels_,
			height_, width_, pooled_height_, pooled_width_, kernel_h_,
			kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
		break;
	case AVE:
		PoolingLayer::AvePoolForward_gpu(
			count, bottom_data, batchSize, channels_,
			height_, width_, pooled_height_, pooled_width_, kernel_h_,
			kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
		break;
	}
}

size_t PoolingLayer::getSerializationSize()
{
    return (13) * sizeof(int) + 1;
}

void PoolingLayer::serialize(void* buffer)
{
	char* d = reinterpret_cast<char*>(buffer);
	d[0] = global_pooling_;
	int* dd = reinterpret_cast<int*>(d);
	dd[1] = kernel_h_;
	dd[2] = kernel_w_;
	dd[3] = stride_h_;
	dd[4] = stride_w_;
	dd[5] = pad_h_;
	dd[6] = pad_w_;
	dd[7] = pooled_height_;
	dd[8] = pooled_width_;
	dd[9] = channels_;
	dd[10] = height_;
	dd[11] = width_;
	dd[12] = round_mode_;
	dd[13] = pool_method_;
}

void PoolingLayer::configure(const nvinfer1::Dims*inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOutputs, int)
{

}