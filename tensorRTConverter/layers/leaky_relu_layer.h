#ifndef __LEAK_RELU_LAYER_H__
#define __LEAK_RELU_LAYER_H__
#include "base_plugin.h"
class LeakyReluLayer
{
public:
    void forward_gpu(const int size,const float *inputs,float *outputs, float negative_slope);
};
class LeakyReluPlugin : public BasePlugin
{
public:
    LeakyReluPlugin()
    {
    }
    LeakyReluPlugin(const void *data, size_t length)
    {
        const char *d = static_cast<const char *>(data), *a = d;
        read(d, input_channels_);
        read(d, input_height_);
        read(d, input_width_);
        assert(d == a + length);
    }

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override
    {
        input_channels_ = inputs[0].d[0];
        input_height_ = inputs[0].d[1];
        input_width_ = inputs[0].d[2];
        return Dims3(input_channels_, input_height_, input_width_);
    }
    const char *getPluginType() const override { return "LeakReluPlugin"; }
    const char *getPluginVersion() const override { return "1.0.0"; }

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override
    {
        //const float *in=reinterpret_cast<const float*>(inputs[0]);
        LeakyReluLayer layer;
        const float *bottom_data = reinterpret_cast<const float *>(inputs[0]);
        float *output = reinterpret_cast<float *>(outputs[0]);
        layer.forward_gpu(input_channels_ * input_height_ * input_width_, bottom_data, output, 0.1);
        return 0;
    }

    size_t getSerializationSize() override
    {
        return sizeof(input_channels_) + sizeof(input_height_) + sizeof(input_width_);
    }

    void serialize(void *buffer) override
    {
        char *d = static_cast<char *>(buffer), *a = d;

        write(d, input_channels_);
        write(d, input_height_);
        write(d, input_width_);
        assert(d == a + getSerializationSize());
    }
};

#endif