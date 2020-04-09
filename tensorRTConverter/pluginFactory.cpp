#include "pluginFactory.h"
#include <regex>
using namespace nvinfer1;
using namespace nvcaffeparser1;

/******************************/
// PluginFactory
/******************************/
nvinfer1::IPlugin *PluginFactory::createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights)
{
    assert(isPlugin(layerName));
    //apollo yolo
    if (!strcmp(layerName, "slice"))
    {
        assert(slice_layer_.get() == nullptr);
        std::vector<int> slice_points{64,80};
        slice_layer_ = std::unique_ptr<SlicePlugin>(new SlicePlugin(slice_points,3));
        return slice_layer_.get();
    }
    //yolov3
    else if (std::regex_match(layerName, std::regex("LeakyReLU_\\d{1,}")))
    {
        //assert(leaky_relu_layer_[i].get() == nullptr);
        leaky_relu_layer_.emplace_back(std::unique_ptr<LeakyReluPlugin>(new LeakyReluPlugin()));
        return leaky_relu_layer_.back().get();
    }
    else if (std::regex_match(layerName, std::regex("UpsamplingBilinear2d_\\d{1,}")))
    {
        upsample_layer_.emplace_back(std::unique_ptr<UpsamplePlugin>(new UpsamplePlugin(2)));
        return upsample_layer_.back().get();
    }
    //rfcn
    else if (!strcmp(layerName, "proposal"))
    {
        assert(proposal_layer_.get() == nullptr);
        proposal_layer_ = std::unique_ptr<ProposalPlugin>(new ProposalPlugin());
        return proposal_layer_.get();
    }
    else if (!strcmp(layerName, "psroipooled_loc_rois"))
    {
        assert(psroi_bbox_.get() == nullptr);
        int pooled_h = 7;
        int pooled_w = 7;
        int output_dim = 8;
        float spatial_sacle = 0.0625;
        psroi_bbox_ = std::unique_ptr<PsroiPoolingPlugin>(new PsroiPoolingPlugin(pooled_h, pooled_w, output_dim, spatial_sacle));
        return psroi_bbox_.get();
    }
    else if (!strcmp(layerName, "psroipooled_cls_rois"))
    {
        assert(psroi_cls_.get() == nullptr);
        int pooled_h = 7;
        int pooled_w = 7;
        int output_dim = 5;
        float spatial_sacle = 0.0625;
        psroi_cls_ = std::unique_ptr<PsroiPoolingPlugin>(new PsroiPoolingPlugin(pooled_h, pooled_w, output_dim, spatial_sacle));
        return psroi_cls_.get();
    }
    else
    {
        std::cout << "not found  " << layerName << std::endl;
        return nullptr;
    }
}

IPlugin *PluginFactory::createPlugin(const char *layerName, const void *serialData, size_t serialLength)
{
    assert(isPlugin(layerName));
    //apollo yolo 
    if (!strcmp(layerName, "slice"))
    {

        assert(slice_layer_.get() == nullptr);
       // slice_layer_ = std::unique_ptr<SlicePlugin>(new SlicePlugin(serialData,serialLength));
        //return slice_layer_.get();
        return new SlicePlugin(serialData,serialLength);
    }
    //yolov3
    if (std::regex_match(layerName, std::regex("LeakyReLU_\\d{1,}")))
    {
        leaky_relu_layer_.emplace_back(std::unique_ptr<LeakyReluPlugin>(new LeakyReluPlugin(serialData, serialLength)));
        return leaky_relu_layer_.back().get();
    }
    else if (std::regex_match(layerName, std::regex("UpsamplingBilinear2d_\\d{1,}")))
    {
        upsample_layer_.emplace_back(std::unique_ptr<UpsamplePlugin>(new UpsamplePlugin(serialData, serialLength)));
        return upsample_layer_.back().get();
    }
    //rfcn
    else if (!strcmp(layerName, "proposal"))
    {
        assert(proposal_layer_.get() == nullptr);
        return new ProposalPlugin(serialData, serialLength);
    }
    else if (!strcmp(layerName, "psroipooled_loc_rois"))
    {
        assert(psroi_bbox_.get() == nullptr);
        return new PsroiPoolingPlugin(serialData, serialLength);
    }
    else if (!strcmp(layerName, "psroipooled_cls_rois"))
    {
        assert(psroi_cls_.get() == nullptr);
        return new PsroiPoolingPlugin(serialData, serialLength);
    }
    else
    {
        return nullptr;
    }
}
bool PluginFactory::isPlugin(const char *name)
{
    return isPluginExt(name);
}
bool PluginFactory::isPluginExt(const char *name)
{
    return
        //apollo yolo
        !strcmp(name, "slice") ||
        //yolov3
        std::regex_match(name, std::regex("LeakyReLU_\\d{1,}")) ||
        std::regex_match(name, std::regex("UpsamplingBilinear2d_\\d{1,}")) ||
        //rfcn
        !strcmp(name, "proposal") ||
        !strcmp(name, "psroipooled_loc_rois") ||
        !strcmp(name, "psroipooled_cls_rois");
}
