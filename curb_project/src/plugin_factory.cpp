#include "plugin_factory.h"

bool PluginFactory::isPlugin(const char* name)
{
    std::string strName {name};
    std::transform(strName.begin(),strName.end(),strName.begin(),::tolower);
    return(strName.find("split1") != std::string::npos) ||
            (strName.find("split2") != std::string::npos) ||
            (strName.find("max_pool11") != std::string::npos);
}

nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights){
    assert(isPlugin(layerName));

    std::string strName {layerName};
    std::transform(strName.begin(),strName.end(),strName.begin(),::tolower);

    if (strName.find("split1") != std::string::npos){
    	vector<int> slicepoint;
    	for(int i=1;i<32;i++)
    	{
    		slicepoint.push_back(i);
    	}
        _nvPlugins[layerName] = (nvinfer1::IPlugin*)(new SliceLayer(slicepoint, 2));
        return _nvPlugins.at(layerName);
    }
    else if (strName.find("split2") != std::string::npos){
        vector<int> slicepoint;
        for(int i=1;i<64;i++)
        {
            slicepoint.push_back(i);
        }
        _nvPlugins[layerName] = (nvinfer1::IPlugin*)(new SliceLayer(slicepoint, 3));
        return _nvPlugins.at(layerName);
    }
    else if (strName.find("max_pool11") != std::string::npos){
        _nvPlugins[layerName] = (nvinfer1::IPlugin*)(new PoolingLayer(3,3,2,2,1,1,0,(PoolingParameter_RoundMode)1,
                                                                                    (PoolingParameter_PoolMethod)0));
        return _nvPlugins.at(layerName);
    }
    else{
        std::cout << "warning : " << layerName << std::endl;
        assert(0);  
        return nullptr;  
    }
}
nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
    assert(isPlugin(layerName));

    std::string strName {layerName};
    std::transform(strName.begin(),strName.end(),strName.begin(),::tolower);

    if (strName.find("split1") != std::string::npos){
        _nvPlugins[layerName] = (nvinfer1::IPlugin*)(new SliceLayer(serialData,serialLength));
        return _nvPlugins.at(layerName);
    }
    else if (strName.find("split2") != std::string::npos){
        _nvPlugins[layerName] = (nvinfer1::IPlugin*)(new SliceLayer(serialData,serialLength));
        return _nvPlugins.at(layerName);
    }
    else if (strName.find("max_pool11") != std::string::npos){
        _nvPlugins[layerName] = (nvinfer1::IPlugin*)(new PoolingLayer(serialData,serialLength));
        return _nvPlugins.at(layerName);
    }
    else{
        std::cout << "warning : " << layerName << std::endl;
        assert(0);  
        return nullptr;  
    }
}

void PluginFactory::destroyPlugin(){
    for (auto it = _nvPlugins.begin(); it!=_nvPlugins.end(); it++){
        if (strstr(it->first.c_str(),"split1") || strstr(it->first.c_str(),"split2")){
            delete (SliceLayer*)(it->second);
            _nvPlugins.erase(it);
        }
        if (strstr(it->first.c_str(),"max_pool11")){
            delete (PoolingLayer*)(it->second);
            _nvPlugins.erase(it);
        }
    }
}