#ifndef __BASER_LAYER_H__
#define __BASER_LAYER_H__
#include "utils.h"
#include "NvInferPlugin.h"
#include <cublas_v2.h>
#include "param.h"
#include "assert.h"
using namespace std;
using namespace nvinfer1;

class BasePlugin : public IPluginExt
{
public:
    BasePlugin() {}
    virtual ~BasePlugin()
    {
    }
    BasePlugin(const void *data, size_t length)
    {
        // const char *d = static_cast<const char *>(data), *a = d;
        // read(d, input_channels_);
        // read(d, input_height_);
        // read(d, input_width_);
        // assert(d == a + length);
    }
    //以下必须重写
    virtual const char *getPluginType() const = 0;
    virtual const char *getPluginVersion() const = 0;
    virtual int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) = 0;
    virtual size_t getSerializationSize() = 0;
    virtual void serialize(void *buffer) override
    {
        // char *d = static_cast<char *>(buffer), *a = d;
        // write(d, input_channels_);
        // write(d, input_height_);
        // write(d, input_width_);
        // assert(d == a + getSerializationSize());
    }
    virtual int getNbOutputs() const override
    {
        return 1;
    }
    virtual Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override
    {
        input_channels_ = inputs[0].d[0];
        input_height_ = inputs[0].d[1];
        input_width_ = inputs[0].d[2];
        return Dims3(input_channels_, input_height_, input_width_);
    }

    //以下可选择性重写
    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        return (type == DataType::kFLOAT ) && format == PluginFormat::kNCHW;
    }

    void configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
    {
        assert((type == DataType::kFLOAT ) && format == PluginFormat::kNCHW);
        mDataType = type;
    }

    int initialize() override
    {
        return 0;
    }

    virtual void terminate() override {}

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        return 0;
    }
    virtual void destroy() {}
    IPluginExt *clone() const { return NULL; }

protected:
    size_t type2size(DataType type) { return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half); }
    template <typename T>
    void write(char *&buffer, const T &val)
    {
        *reinterpret_cast<T *>(buffer) = val;
        buffer += sizeof(T);
    }
    template <typename T>
    void read(const char *&buffer, T &val)
    {
        val = *reinterpret_cast<const T *>(buffer);
        buffer += sizeof(T);
    }
    void *copyToDevice(const void *data, size_t count)
    {
        void *deviceData;
        cudaMalloc(&deviceData, count);
        cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice);
        return deviceData;
    }
    void convertAndCopyToDevice(void *&deviceWeights, const Weights &weights) {}
    void convertAndCopyToBuffer(char *&buffer, const Weights &weights) {}
    void deserializeToDevice(const char *&hostBuffer, void *&deviceWeights, size_t size)
    {
        deviceWeights = copyToDevice(hostBuffer, size);
        hostBuffer += size;
    }
    DataType mDataType{DataType::kFLOAT};
    int input_channels_;
    int input_height_;
    int input_width_;
};

#endif
