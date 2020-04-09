#ifndef __TENSORNET_H__
#define __TENSORNET_H__
#include "pluginFactory.h"
#include <iostream>
using namespace nvinfer1;
using namespace nvcaffeparser1;
/******************************/
// TensorRT utility
/******************************/
class Logger : public ILogger
{
    void log(Severity severity, const char *msg) override
    {
        if (severity != Severity::kINFO)
            ; //std::cout << msg << std::endl;
    }
};

struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char *layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record &r) { return r.first == layerName; });

        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

    void printLayerTimes(const int TIMING_ITERATIONS)
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
    }
};

/******************************/
// TensorRT Main
/******************************/
class TensorNet
{
public:
    bool caffeToTRTModel(const char *deployFile,
                         const char *modelFile,
                         const std::vector<std::string> &outputs,
                         unsigned int maxBatchSize,
                         std::ostream &trtModelStdStream,
                         IInt8Calibrator *calibrator = nullptr);
    bool onnxToTRTModel(const std::string& modelFile, // name of the onnx model
                    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                    std::ostream &trtModelStdStream); // output buffer for the TensorRT model
    bool LoadNetwork(std::string model_folder,
                     std::string prototxt_path,
                     std::string model_path,
                     const std::vector<std::string> &output_blobs,
                     uint32_t maxBatchSize
                  );
    bool LoadNetwork(const char *model_path);

    void createInference();

    void imageInference(void **buffers, int nbBuffer, int batchSize);

    DimsCHW getTensorDims(const char *name);

    //    void getLayerOutput(const char* name);

    void printTimes(int iteration);
    void destroy();
    float *allocateMemory(DimsCHW dims, char *info);

private:
    PluginFactory pluginFactory;
    IHostMemory *trtModelStream{nullptr};

    IRuntime *infer;
    ICudaEngine *engine;

    Logger gLogger;
    Profiler gProfiler;
};

#endif
