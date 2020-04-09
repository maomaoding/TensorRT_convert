#ifndef __PROPOSAL_LAYER_H__
#define __PROPOSAL_LAYER_H__
#include "base_plugin.h"
class ProposalLayer 
{
private:
    /* data */
    float *anchors_ = nullptr;
    float *proposals_ = nullptr;
    vector<int> anchor_shape_;
    int *roi_indices_ = nullptr;
    int *nms_mask_ = nullptr;
    int base_size_;
    int feat_stride_;
    int pre_nms_topn_;
    int post_nms_topn_;
    float nms_thresh_;
    int min_size_;

    vector<int> input_cls_shape_;
    vector<int> input_bbox_shape_;
    vector<int> output_shape_;
    float *rois_ = nullptr;
    float *rois_host_ = nullptr;

public:
    ProposalLayer(void **output, vector<int> input_cls_shape, vector<int> input_bbox_shape, vector<int> output_shape);
    ~ProposalLayer();
    void LayerSetUp();
    void forward_cpu(const void *const *inputs);
    void forward_gpu(const void *const *inputs);
};

class ProposalPlugin : public BasePlugin
{
public:
    ProposalPlugin() {}
    // create the plugin at runtime from a byte stream
    ProposalPlugin(const void *data, size_t length)
    {
        const char *d = static_cast<const char *>(data), *a = d;
        read(d, input_cls_channels_);
        read(d, input_cls_height_);
        read(d, input_cls_width_);
        read(d, input_bbox_channels_);
        read(d, input_bbox_height_);
        read(d, input_bbox_width_);
        assert(d == a + length);
    }

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override
    {
        input_cls_channels_ = inputs[0].d[0];
        input_cls_height_ = inputs[0].d[1];
        input_cls_width_ = inputs[0].d[2];

        input_bbox_channels_ = inputs[1].d[0];
        input_bbox_height_ = inputs[1].d[1];
        input_bbox_width_ = inputs[1].d[2];
        return Dims3(RPN_NMS_MAX, 5, 1);
    }

    const char *getPluginType() const override { return "ProposalPlugin"; }
    const char *getPluginVersion() const override { return "1.0.0"; }

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override
    {
        //const float *in=reinterpret_cast<const float*>(inputs[0]);
        vector<int> input_cls_shape{input_cls_channels_, input_cls_height_, input_cls_width_};
        vector<int> input_bbox_shape{input_bbox_channels_, input_bbox_height_, input_bbox_width_};
        vector<int> output_shape{RPN_NMS_MAX, 5, 1};
        ProposalLayer layer(outputs, input_cls_shape, input_bbox_shape, output_shape);
        layer.forward_gpu(inputs);

        return 0;
    }

    size_t getSerializationSize() override
    {
        return sizeof(input_cls_channels_) + sizeof(input_cls_height_) + sizeof(input_cls_width_) +
               sizeof(input_bbox_channels_) + sizeof(input_bbox_height_) + sizeof(input_bbox_width_);
    }

    void serialize(void *buffer) override
    {
        char *d = static_cast<char *>(buffer), *a = d;

        write(d, input_cls_channels_);
        write(d, input_cls_height_);
        write(d, input_cls_width_);
        write(d, input_bbox_channels_);
        write(d, input_bbox_height_);
        write(d, input_bbox_width_);

        assert(d == a + getSerializationSize());
    }

private:
    int input_cls_channels_;
    int input_cls_height_;
    int input_cls_width_;

    int input_bbox_channels_;
    int input_bbox_height_;
    int input_bbox_width_;
};
#endif
