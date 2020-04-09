#include <algorithm>
#include "tensorNet.h"
#include <sstream>
#include <fstream>

using namespace nvinfer1;
bool TensorNet::LoadNetwork(std::string model_folder,
                            std::string prototxt_path,
                            std::string model_path,
                            const std::vector<std::string>& output_blobs,
                            uint32_t maxBatchSize
                          )
{
    
    std::stringstream trtModelStdStream;
    trtModelStdStream.seekg(0, trtModelStdStream.beg);
    char cache_path[512];
    std::string path=model_folder+"/tensorRT_model.tensorcache";
    sprintf(cache_path, path.c_str());
    printf( "attempting to open cache file %s\n", cache_path);
    if(model_path!="")
    {
        if( !caffeToTRTModel(prototxt_path.c_str(), model_path.c_str(), output_blobs, maxBatchSize, trtModelStdStream) )
        {
            printf("failed to load %s\n", model_path);
            return 0;
        }
    }
    else
    {
         if (!onnxToTRTModel(prototxt_path,maxBatchSize, trtModelStdStream))
         {
            printf("failed to load %s\n", model_path);
            return 0;
        }
    }
	printf( "network profiling complete, writing cache to %s\n", cache_path);
	std::ofstream outFile;
	outFile.open(cache_path);
	outFile << trtModelStdStream.rdbuf();
	outFile.close();
	trtModelStdStream.seekg(0, trtModelStdStream.beg);
	printf( "completed writing cache to %s\n", cache_path);
   
}

bool TensorNet::caffeToTRTModel(const char* deployFile,
                                const char* modelFile,
                                const std::vector<std::string>& outputs,
                                unsigned int maxBatchSize,
                                std::ostream& trtModelStdStream,
                                IInt8Calibrator* calibrator)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    //    builder->setMinFindIterations(3);	// allow time for TX1 GPU to spin up
    //    builder->setAverageFindIterations(2);
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(&pluginFactory);

    bool useFp16 = builder->platformHasFastFp16();
    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT;
    std::cout << deployFile <<std::endl;
    std::cout << modelFile <<std::endl;

	std::cout << "start parseing" <<std::endl;
    const IBlobNameToTensor* blobNameToTensor =	parser->parse(deployFile,
                                                              modelFile,
                                                              *network,
                                                              modelDataType);
    std::cout << "finish parseing" <<std::endl;
    assert(blobNameToTensor != nullptr);
    for (auto& s : outputs) {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }
    
    builder->setMaxWorkspaceSize(1 << 30);
    builder->setDebugSync(false);
    builder->setMaxBatchSize(maxBatchSize);
    if(useFp16 && !INT8_MODE)
    {
        builder->setHalf2Mode(true);
    }
    std::cout << "start engine building" <<std::endl;
    ICudaEngine* engine = builder->buildCudaEngine( *network );
    std::cout << "finish engine building" <<std::endl;

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();
    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    if(!trtModelStream)
    {
        std::cout << "failed to serialize CUDA engine" << std::endl;
        return false;
    }
    trtModelStdStream.write((const char*)trtModelStream->data(),trtModelStream->size());
    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
    std::cout << "caffeToTRTModel Finished" << std::endl;
    return true;
}
bool TensorNet::onnxToTRTModel(const std::string& modelFile, // name of the onnx model
                    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                    std::ostream &trtModelStdStream) // output buffer for the TensorRT model
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    assert(builder != nullptr);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    auto parser = nvonnxparser::createParser(*network,gLogger);

    //Optional - uncomment below lines to view network layer information
    //config->setPrintLayerInfo(true);
    //parser->reportParsingInfo();
    if ( !parser->parseFromFile(modelFile.c_str(), 0 ) )
    {
        std::cout << "Failure while parsing ONNX file" << std::endl;
        return false;
    }
    bool useFp16 = builder->platformHasFastFp16();
    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT;
    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    builder->setFp16Mode(useFp16);
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we can destroy the parser
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    if(!trtModelStream)
    {
        std::cout << "failed to serialize CUDA engine" << std::endl;
        return false;
    }
    trtModelStdStream.write((const char*)trtModelStream->data(),trtModelStream->size());
    engine->destroy();
    network->destroy();
    builder->destroy();

    return true;
}
void TensorNet::destroy()
{
    trtModelStream->destroy();
    pluginFactory.destroyPlugin();
}
/**
 * This function de-serializes the cuda engine.
 * */
void TensorNet::createInference()
{
    infer = createInferRuntime(gLogger);
    /**
     * deserializeCudaEngine can be used to load the serialized CuDA Engine (Plan file).
     * */
    std::cout << "deserialize engine" << std::endl;
    engine = infer->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), &pluginFactory);

    printf("Bindings after deserializing:\n");
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
        if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
        else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
    }
}

void TensorNet::imageInference(void** buffers, int nbBuffer, int batchSize)
{
    //std::cout << "Came into the image inference method here. "<<std::endl;
    assert( engine->getNbBindings()==nbBuffer);
    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);
    context->execute(batchSize, buffers);
    //std::cout<<111<<std::endl;
    context->destroy();
}


DimsCHW TensorNet::getTensorDims(const char* name)
{
    for (int b = 0; b < engine->getNbBindings(); b++) {
        if( !strcmp( name, engine->getBindingName(b)) )
            return static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
    }
    return DimsCHW{0,0,0};
}



float* TensorNet::allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = 1 * dims.c() * dims.h() * dims.w();
    //std::cout<< dims.c() <<" "<<dims.h()<<" "<< dims.w()<<std::endl;
    CHECK(cudaMallocManaged( &ptr, size*sizeof(float)));
    return ptr;
}
