#ifndef SLICE_LAYER_HPP_
#define SLICE_LAYER_HPP_

#include "utils.h"

class SliceLayer : public nvinfer1::IPlugin
{
public:
	SliceLayer(const std::vector<int>& slice_point, int slice_axis);
	SliceLayer(const void* buffer, size_t size);
	inline int getNbOutputs() const override { return top_slice_axis_.size(); };
	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;
	int initialize() override{ return 0; };
	inline void terminate() override{}
	inline size_t getWorkspaceSize(int) const override { return 0; }
	int enqueue(int batchSize, const void* const *inputs, void** outputs, void*, cudaStream_t stream) override;
	size_t getSerializationSize() override;
	void serialize(void* buffer) override;
	void configure(const nvinfer1::Dims*inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOutputs, int) override;
	void slicelayer_gpu(const int nthreads, const float* in_data,
	    const bool forward, const int num_slices, const int slice_size,
	    const int bottom_slice_axis, const int top_slice_axis,
	    const int offset_slice_axis, float* out_data);

private:
	int channel_;               //特征图通道数
	int slice_size_;            //(slice_axis_+1,...)维度连乘
	int slice_axis_;            //slice的坐标维度
	int bottom_slice_axis_;     //上一层slice的坐标维度
	vector<int> slice_point_;
	vector<int> top_slice_axis_;//输出每个top的slice坐标的维度大小
};

#endif