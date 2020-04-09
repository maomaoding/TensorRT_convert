#ifndef PLUGINFACTORY_HPP_
#define PLUGINFACTORY_HPP_
#include "utils.h"
#include "slice_layer.h"
#include "pooling_layer.h"

class PluginFactory: public nvinfer1::IPluginFactory,
                      public nvcaffeparser1::IPluginFactory {
public:
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override  ;
	nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override ; 
    bool isPlugin(const char* name) override;
    void destroyPlugin();
private:
    std::map<std::string, nvinfer1::IPlugin* > _nvPlugins; 
};

#endif