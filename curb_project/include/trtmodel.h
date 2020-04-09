#ifndef TRTMODEL_HPP_
#define TRTMODEL_HPP_

#include "plugin_factory.h"

class TRTmodel
{
public:
	TRTmodel(string deployFile, string modelFile, string tensorrt_cache):
			deployFile(deployFile), modelFile(modelFile), tensorrt_cache(tensorrt_cache)
	{
		std::shared_ptr<char> engine_buffer;
		int engine_buffer_size;

		TRTmodel::ReadModel(engine_buffer, engine_buffer_size);

		runtime = nvinfer1::createInferRuntime(gLogger);
		assert(runtime != nullptr);

		engine = runtime->deserializeCudaEngine(engine_buffer.get(),
												engine_buffer_size,
												&pluginFactory);
		assert(engine != nullptr);

		context = engine->createExecutionContext();
		assert(context != nullptr);
	}

	~TRTmodel()
	{
		if(context)
			context->destroy();
		if(engine)
			engine->destroy();
		if(runtime)
			runtime->destroy();
		pluginFactory.destroyPlugin();
	}

	void caffeToTRTModel();

	void doInference(float* input, float* output, int batchSize);

	void ReadModel(std::shared_ptr<char>& engine_buffer,
					 int& engine_buffer_size);

	nvinfer1::DimsCHW getTensorDims(const char* TensorName);

private:
	Logger gLogger;
	string deployFile;
	string modelFile;
	string tensorrt_cache;
	unsigned int maxBatchSize = 1;
	PluginFactory pluginFactory;

	nvinfer1::ICudaEngine* engine{ nullptr };
	nvinfer1::IRuntime* runtime{ nullptr };
	nvinfer1::IExecutionContext *context{ nullptr };

#ifdef denseaspp
	const char* INPUT_BLOB_NAME = "blob1";
	const char* OUTPUT_BLOB_NAME = "conv_blob132";
#endif
#ifdef scnncaffe_udlr
	const char* INPUT_BLOB_NAME = "data";
	const char* OUTPUT_BLOB_NAME = "softmax";
#endif
#ifdef vggscnn
	const char* INPUT_BLOB_NAME = "blob1";
	const char* OUTPUT_BLOB_NAME = "conv_blob203";
#endif

};

#endif