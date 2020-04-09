#include "trtmodel.h"

nvinfer1::DimsCHW TRTmodel::getTensorDims(const char* TensorName)
{
    assert(engine != nullptr);

    return static_cast<nvinfer1::DimsCHW&&>(engine->getBindingDimensions(engine->getBindingIndex(TensorName)));
}

void TRTmodel::caffeToTRTModel()
{
	// create the builder
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
	
	// parse the caffe model to populate the network, then set the outputs
	nvinfer1::INetworkDefinition* network = builder->createNetwork();
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
	parser->setPluginFactory(&pluginFactory);

	bool fp16 = builder->platformHasFastFp16();

	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
															modelFile.c_str(),
															*network, fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
    // specify which tensors are outputs
	const std::vector<std::string>& outputs = { OUTPUT_BLOB_NAME };
	for(auto& s:outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);
	builder->setFp16Mode(fp16);

	engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

	// serialize the engine, then close everything down
    nvinfer1::IHostMemory *trtModelStream{ nullptr };
    trtModelStream = engine->serialize();

    engine->destroy();
    builder->destroy();
    nvcaffeparser1::shutdownProtobufLibrary();

    cout<<"RT init done!"<<endl;

    ofstream out(tensorrt_cache.c_str(),ios::out|ios::binary);
    out.write((const char*)(trtModelStream->data()),trtModelStream->size());
    out.close();

    pluginFactory.destroyPlugin();
    assert(trtModelStream != nullptr);
    trtModelStream->destroy();
}

void TRTmodel::doInference(float* input, float* output, int batchSize)
{
    // std::shared_ptr<char> engine_buffer;
    // int engine_buffer_size;

    // TRTmodel::ReadModel(engine_buffer, engine_buffer_size);

    // runtime = nvinfer1::createInferRuntime(gLogger);
    // assert(runtime != nullptr);

    // engine = runtime->deserializeCudaEngine(engine_buffer.get(),
    //                                         engine_buffer_size,
    //                                         &pluginFactory);
    assert(engine != nullptr);

    // context = engine->createExecutionContext();
    assert(context != nullptr);

    // const nvinfer1::ICudaEngine& engine = context->getEngine();
    // assert(engine.getNbBindings() == 2);

    void* buffers[2];

    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME),
        outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

    nvinfer1::DimsCHW dim_in = getTensorDims(INPUT_BLOB_NAME);
    nvinfer1::DimsCHW dim_out = getTensorDims(OUTPUT_BLOB_NAME);

    cudaMalloc(&buffers[inputIndex], batchSize * dim_in.h() * dim_in.w() * dim_in.c() * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batchSize * dim_out.h() * dim_out.w() * dim_out.c() * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(buffers[inputIndex], input, batchSize * dim_in.h() * dim_in.w() * dim_in.c() * sizeof(float), cudaMemcpyHostToDevice, stream);

    boost::timer t;
    context->enqueue(batchSize, buffers, stream, nullptr);
    cout<<"enqueue time:"<<t.elapsed()<<endl;

    cudaMemcpyAsync(output, buffers[outputIndex], batchSize * dim_out.h() * dim_out.w() * dim_out.c() * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}

void TRTmodel::ReadModel(std::shared_ptr<char>& engine_buffer, int& engine_buffer_size){
    std::ifstream in(tensorrt_cache.c_str(),std::ios::in | std::ios::binary);
    if (!in.is_open()){
        engine_buffer_size = 0;
        engine_buffer = nullptr;
    }

    in.seekg(0,std::ios::end);
    engine_buffer_size = in.tellg();
    in.seekg(0,std::ios::beg);
    engine_buffer.reset(new char[engine_buffer_size]);
    in.read(engine_buffer.get(),engine_buffer_size);
    in.close();
}