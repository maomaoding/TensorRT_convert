#ifndef __UPSAMPLE_LAYER_H__
#define __UPSAMPLE_LAYER_H__
#include "base_plugin.h"
class UpsampleLayer 
{
public:
    void forward_gpu(const float *input, float *output, float scale, int N, int C, int H, int W);
};

class UpsamplePlugin : public BasePlugin
{
public:
    UpsamplePlugin(const float scale)
        : scale_(scale)
    {
    }
    UpsamplePlugin(const void *data, size_t length)
    {
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, input_channels_);
        read(d, input_height_);
        read(d, input_width_);
        read(d, output_channels_);
        read(d, output_height_);
        read(d, output_width_);
        read(d, scale_);
        //std::cout << "read:" << a << " " << mOutputWidth<< " " <<mOutputHeight<<std::endl;
        assert(d == a + length);
    }

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override
    {
        input_channels_ = inputs[0].d[0];
        input_height_ = inputs[0].d[1];
        input_width_ = inputs[0].d[2];
        output_channels_ = input_channels_;
        output_height_ = input_height_ * scale_;
        output_width_ = input_width_ * scale_;
        return Dims3(input_channels_, input_height_ * scale_, input_width_ * scale_);
    }
    const char *getPluginType() const override { return "UpsamplePlugin"; }
    const char *getPluginVersion() const override { return "1.0.0"; }
    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override
    {
        //const float *in=reinterpret_cast<const float*>(inputs[0]);
        UpsampleLayer layer;
        layer.forward_gpu((const float *)inputs[0], (float *)outputs[0], scale_, batchSize, output_channels_, output_height_, output_width_);
        return 0;
    }

    size_t getSerializationSize() override
    {
        return sizeof(input_channels_) + sizeof(input_height_) + sizeof(input_width_) +
               sizeof(output_channels_) + sizeof(output_height_) + sizeof(output_width_) +
               sizeof(scale_);
    }

    void serialize(void *buffer) override
    {
        char *d = static_cast<char *>(buffer), *a = d;

        write(d, input_channels_);
        write(d, input_height_);
        write(d, input_width_);

        write(d, output_channels_);
        write(d, output_height_);
        write(d, output_width_);

        write(d, scale_);
        assert(d == a + getSerializationSize());
    }

private:
    float scale_;
    int output_channels_;
    int output_height_;
    int output_width_;
};

#endif