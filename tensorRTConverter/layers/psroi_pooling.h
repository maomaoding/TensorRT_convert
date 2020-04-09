
#ifndef __PSROI_LAYER_H__
#define __PSROI_LAYER_H__
#include "base_plugin.h"
class PsroiPooling 
{
private:
    //float *bottom_data_=nullptr;
    //float *bottom_roi_=nullptr;
    float *top_data_ = nullptr;

    vector<int> input_featrue_shape_;
    vector<int> input_roi_shape_;
    vector<int> output_shape_;
    float spatial_scale_;

public:
    PsroiPooling(void **output, vector<int> input_feature_shape, vector<int> input_roi_shape, vector<int> output_shape, float spatial_sacle);
    void forward_gpu(const void *const *inputs);
};

class PsroiPoolingPlugin : public BasePlugin
{
public:
    PsroiPoolingPlugin(int pooled_h, int pooled_w, int output_dim, float spatial_sacle) : pooled_h_(pooled_h), pooled_w_(pooled_w),
                                                                                          output_dim_(output_dim), spatial_sacle_(spatial_sacle)
    {
    }

    // create the plugin at runtime from a byte stream
    PsroiPoolingPlugin(const void *data, size_t length)
    {

        const char *d = static_cast<const char *>(data), *a = d;
        read(d, input_feature_channels_);
        read(d, input_feature_height_);
        read(d, input_feature_width_);
        read(d, input_roi_channels_);
        read(d, input_roi_height_);
        read(d, input_roi_width_);
        read(d, pooled_h_);
        read(d, pooled_w_);
        read(d, output_dim_);
        read(d, spatial_sacle_);
        assert(d == a + length);
    }

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override
    {
        input_feature_channels_ = inputs[0].d[0];
        input_feature_height_ = inputs[0].d[1];
        input_feature_width_ = inputs[0].d[2];

        input_roi_channels_ = inputs[1].d[0];
        input_roi_height_ = inputs[1].d[1];
        input_roi_width_ = inputs[1].d[2];
        //std::cout<<input_feature_channels_<< " "<<input_feature_height_<<" "<<input_feature_width_<<std::endl;
        //std::cout<<input_roi_channels_<< " "<<input_roi_height_<<" "<<input_roi_width_<<std::endl;
        return Dims3(output_dim_ * input_roi_channels_, pooled_h_, pooled_w_);
    }

    const char *getPluginType() const override { return "PsroiPoolingPlugin"; }
    const char *getPluginVersion() const override { return "1.0.0"; }

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override
    {
        //const float *in=reinterpret_cast<const float*>(inputs[0]);
        vector<int> input_featrue_shape{input_feature_channels_, input_feature_height_, input_feature_width_};
        vector<int> input_roi_shape{input_roi_channels_, input_roi_height_, input_roi_width_};
        vector<int> output_shape{output_dim_, pooled_h_, pooled_w_};
        PsroiPooling layer(outputs, input_featrue_shape, input_roi_shape, output_shape, spatial_sacle_);
        layer.forward_gpu(inputs);

        return 0;
    }

    size_t getSerializationSize() override
    {
        return sizeof(input_feature_channels_) + sizeof(input_feature_height_) + sizeof(input_feature_width_) +
               sizeof(input_roi_channels_) + sizeof(input_roi_height_) + sizeof(input_roi_width_) + sizeof(pooled_h_) +
               sizeof(pooled_w_) + sizeof(output_dim_) + sizeof(spatial_sacle_);
    }

    void serialize(void *buffer) override
    {
        char *d = static_cast<char *>(buffer), *a = d;

        write(d, input_feature_channels_);
        write(d, input_feature_height_);
        write(d, input_feature_width_);
        write(d, input_roi_channels_);
        write(d, input_roi_height_);
        write(d, input_roi_width_);
        write(d, pooled_h_);
        write(d, pooled_w_);
        write(d, output_dim_);
        write(d, spatial_sacle_);
        assert(d == a + getSerializationSize());
    }

private:
    int pooled_h_;
    int pooled_w_;
    int output_dim_;
    float spatial_sacle_;

    int input_feature_channels_;
    int input_feature_height_;
    int input_feature_width_;

    int input_roi_channels_;
    int input_roi_height_;
    int input_roi_width_;
};
#endif
