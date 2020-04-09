#include "slice_layer.h"

SliceLayer::SliceLayer(const std::vector<int>& slice_point, int slice_axis):
															slice_point_(slice_point),
															slice_axis_(slice_axis)
															// bottom_slice_axis_(bottom_slice_axis)
{
    if(slice_axis_ == 2)
    {
        slice_size_ = 64;
        bottom_slice_axis_ = 32;
        channel_ = 128;
        int prev = 0;
        for (int i = 0; i < slice_point_.size(); ++i)
        {
            top_slice_axis_.push_back(slice_point_[i] - prev);
            prev = slice_point_[i];
        }
        top_slice_axis_.push_back(bottom_slice_axis_ - prev);
    }
    else if(slice_axis_ == 3)
    {
        slice_size_ = 1;
        bottom_slice_axis_ = 64;
        channel_ = 128;
        int prev = 0;
        for (int i = 0; i < slice_point_.size(); ++i)
        {
            top_slice_axis_.push_back(slice_point_[i] - prev);
            prev = slice_point_[i];
        }
        top_slice_axis_.push_back(bottom_slice_axis_ - prev);
    }
}

SliceLayer::SliceLayer(const void* buffer,size_t size)
{
    const int* d = reinterpret_cast<const int*>(buffer);
    int total_lenth = size / sizeof(int);
    for(int i=0;i<total_lenth;i++)
    {
    	if(i<4)
    	{
    		channel_ = d[0];
    		slice_size_ = d[1];
    		slice_axis_ = d[2];
    		bottom_slice_axis_ = d[3];
    	}else{
    		slice_point_.push_back(d[i]);
    	}
    }

	int prev = 0;
    for (int i = 0; i < slice_point_.size(); ++i)
    {
		top_slice_axis_.push_back(slice_point_[i] - prev);
		prev = slice_point_[i];
    }
    top_slice_axis_.push_back(bottom_slice_axis_ - prev);
}

nvinfer1::Dims SliceLayer::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
	if(slice_axis_ == 2)
	{
		return nvinfer1::DimsCHW(inputs[0].d[0], top_slice_axis_[index], inputs[0].d[2]);
	}else if(slice_axis_ == 3){
		return nvinfer1::DimsCHW(inputs[0].d[0], inputs[0].d[1], top_slice_axis_[index]);
	}
}

int SliceLayer::enqueue(int batchSize, const void* const *inputs, void** outputs, void*, 
																	cudaStream_t stream)
{
	int offset_slice_axis = 0;
	const float* bottom_data = reinterpret_cast<const float*>(inputs[0]);
	const bool kForward = true;
	// int num_slices_ = batchSize * channel_;
    int num_slices_ = slice_axis_ == 2 ? batchSize * channel_ : batchSize * channel_*32;
	for(int i=0; i<top_slice_axis_.size(); i++)
	{
		float* top_data = reinterpret_cast<float*>(outputs[i]);
		const int top_slice_size = top_slice_axis_[i] * slice_size_;
		const int nthreads = top_slice_size * num_slices_;
    	SliceLayer::slicelayer_gpu(nthreads, bottom_data, kForward, num_slices_, slice_size_,
                    			bottom_slice_axis_, top_slice_axis_[i], offset_slice_axis,
                    			top_data);
        offset_slice_axis += top_slice_axis_[i];
	}
    return 0;
}

size_t SliceLayer::getSerializationSize()
{
    return (4+slice_point_.size()) * sizeof(int);
}

void SliceLayer::serialize(void* buffer)
{
    int* d = reinterpret_cast<int*>(buffer);
    d[0] = channel_;
    d[1] = slice_size_;
    d[2] = slice_axis_;
    d[3] = bottom_slice_axis_;
    for(int i=0;i<slice_point_.size(); i++)
    {
    	d[4+i] = slice_point_[i];
    }
}

void SliceLayer::configure(const nvinfer1::Dims*inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOutputs, int)
{
  //   bottom_slice_axis_ = inputs[0].d[1];
  //   channel_ = inputs[0].d[0];
  //  	int prev = 0;
  //   for (int i = 0; i < slice_point_.size(); ++i)
  //   {
		// top_slice_axis_.push_back(slice_point_[i] - prev);
		// prev = slice_point_[i];
  //   }
  //   top_slice_axis_.push_back(bottom_slice_axis_ - prev);
}