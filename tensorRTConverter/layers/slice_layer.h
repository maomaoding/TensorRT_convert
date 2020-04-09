#ifndef __SLICE_LAYER_H__
#define __SLICE_LAYER_H__
#include "base_plugin.h"
class SliceLayer
{
public:
    SliceLayer(){}
    ~SliceLayer(){}
    void forward_gpu(const float *in_data, void **out_data, const int num_slices, const int slice_size, int slice_axis, vector<int> bottom_shape, vector<int> output_shape, int output_shape_size);
};

class SlicePlugin : public BasePlugin
{
public:
    SlicePlugin(std::vector<int> slice_points, int slice_axis) : slice_points_(slice_points), slice_points_size_(slice_points.size()), slice_axis_(slice_axis) {}
    SlicePlugin(const void *data, size_t length)
    {
        const char* d = static_cast<const char*>(data), *a = d;
        read(d, input_channels_);
        read(d, input_height_);
        read(d, input_width_);
        read(d, slice_axis_);
        read(d, slice_size_);
        read(d, num_slices_);
        read(d, output_shape_size_);
        output_shape_.resize(output_shape_size_);
        for(int i=0;i<output_shape_size_;i++){
             read(d, output_shape_[i]);
        }
        assert(d == a + length);
    }
    int getNbOutputs() const override
    {
        return slice_points_size_+1;
    }
    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override
    {
        input_channels_ = inputs[0].d[0];
        input_height_ = inputs[0].d[1];
        input_width_ = inputs[0].d[2];

        vector<int> input_shape{1, input_channels_, input_height_, input_width_};
        Dims3 output_shape{input_channels_, input_height_, input_width_};
        num_slices_ = 1;
        for (int i = 0; i < slice_axis_; i++)
        {
            num_slices_ *= input_shape[i];
        }

        slice_size_ = 1;
        for (int i = slice_axis_ + 1; i < 4; i++)
        {
            slice_size_ *= input_shape[i];
        }
        int bottom_slice_axis = input_shape[slice_axis_];
        output_shape_.clear();
        int prev = 0;
        for (int i = 0; i < slice_points_size_; ++i)
        {
            output_shape_.push_back(slice_points_[i] - prev);
            prev = slice_points_[i];
        }
        output_shape_.push_back(bottom_slice_axis - prev);

        output_shape_size_ = output_shape_.size();
        for (int i = 0; i < slice_points_size_ + 1; ++i)
        {
            if (index == i)
            {
                output_shape.d[slice_axis_ - 1] = output_shape_[i];
                return output_shape;
            }
        }
    }
    const char *getPluginType() const override { return "SlicePlugin"; }
    const char *getPluginVersion() const override { return "1.0.0"; }

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override
    {
        vector<int> input_shape{1, input_channels_, input_height_, input_width_};
        SliceLayer slicer_layer;
        slicer_layer.forward_gpu(reinterpret_cast<const float *>(inputs[0]), outputs, num_slices_, slice_size_, slice_axis_, input_shape, output_shape_, output_shape_size_);
        return 0;
    }

    size_t getSerializationSize() override
    {
        return sizeof(input_channels_) + sizeof(input_height_) + sizeof(input_width_) + sizeof(slice_axis_) + sizeof(slice_size_) + sizeof(num_slices_) + sizeof(output_shape_size_) + sizeof(int) * output_shape_size_;
    }

    void serialize(void *buffer) override
    {
        char *d = static_cast<char *>(buffer), *a = d;

        write(d, input_channels_);
        write(d, input_height_);
        write(d, input_width_);
        write(d, slice_axis_);
        write(d, slice_size_);
        write(d, num_slices_);
        write(d, output_shape_size_);
        for (int i = 0; i < output_shape_size_; i++)
        {
            write(d, output_shape_[i]);
        }
        assert(d == a + getSerializationSize());
    }

private:
    std::vector<int> output_shape_;
    int output_shape_size_;
    std::vector<int> slice_points_;
    int slice_points_size_;
    int slice_axis_;
    int num_slices_;
    int slice_size_;
};
#endif