#ifndef __PLUGIN_LAYER_H__
#define __PLUGIN_LAYER_H__
#include "NvCaffeParser.h"
#include "NvOnnxParser.h"
#include "utils.h"
#include "param.h"
#include <iostream>
//apollo yolo
#include "layers/slice_layer.h"
//yolov3
#include "layers/leaky_relu_layer.h"
#include "layers/upsample_layer.h"
//rfcn
#include "layers/proposal_layer.h"
#include "layers/psroi_pooling.h"
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
private:
    //根据自定义层添加
    std::vector<std::unique_ptr<LeakyReluPlugin>> leaky_relu_layer_;
    std::vector<std::unique_ptr<UpsamplePlugin>> upsample_layer_;
    std::unique_ptr<ProposalPlugin> proposal_layer_{nullptr};
    std::unique_ptr<PsroiPoolingPlugin> psroi_cls_{nullptr};
    std::unique_ptr<PsroiPoolingPlugin> psroi_bbox_{nullptr};
    std::unique_ptr<SlicePlugin> slice_layer_{ nullptr };

public:
    bool isPlugin(const char *name);
    bool isPluginExt(const char *name);
    void destroyPlugin()
    {
    }
    virtual nvinfer1::IPlugin *createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights) override;
    IPlugin *createPlugin(const char *layerName, const void *serialData, size_t serialLength) override;
    void (*nvPluginDeleter)(INvPlugin *){[](INvPlugin *ptr) { ptr->destroy(); }};
};
#endif
