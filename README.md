# TensorRT convertor(for caffemodel or onnx)

```
├── curb_project
│   ├── 6mm.ini
│   ├── CMakeLists.txt
│   ├── include
│   │   ├── camera_base.h
│   │   ├── common.h
│   │   ├── detect3d.h
│   │   ├── imgskeletonization.h
│   │   ├── NvCaffeParser.h
│   │   ├── NvInfer.h
│   │   ├── NvInferPlugin.h
│   │   ├── NvOnnxConfig.h
│   │   ├── NvOnnxParser.h
│   │   ├── NvUtils.h
│   │   ├── plugin_factory.h
│   │   ├── pooling_layer.h
│   │   ├── slice_layer.h
│   │   ├── trtmodel.h
│   │   └── utils.h
│   ├── launch
│   │   └── start_curb_perception.launch
│   ├── main.cpp
│   ├── package.xml
│   └── src
│       ├── camera_base.cpp
│       ├── detect3d.cpp
│       ├── plugin_factory.cpp
│       ├── pooling_layer.cpp
│       ├── pooling_layer.cu
│       ├── slice_layer.cpp
│       ├── slice_layer.cu
│       ├── trtmodel.cpp
│       └── utils
│           ├── imgskeletonization.cpp
│           └── utils.cpp
├── README.md
├── tensorRTConverter
│   ├── buildTensorRTModel.cpp
│   ├── CMakeLists.txt
│   ├── layers
│   │   ├── base_plugin.h
│   │   ├── leaky_relu_layer.cu
│   │   ├── leaky_relu_layer.h
│   │   ├── proposal_layer.cpp
│   │   ├── proposal_layer.cu
│   │   ├── proposal_layer.h
│   │   ├── psroi_pooling.cu
│   │   ├── psroi_pooling.h
│   │   ├── slice_layer.cu
│   │   ├── slice_layer.h
│   │   ├── upsample_layer.cu
│   │   └── upsample_layer.h
│   ├── param.h
│   ├── pluginFactory.cpp
│   ├── pluginFactory.h
│   ├── tensorNet.cpp
│   ├── tensorNet.h
│   ├── utils.cpp
│   ├── utils.cu
│   └── utils.h
```

## tensorRT5 convertor tool
support centernet obj det
support unetpp lane det		
support rfcn tl det	

## Usage
```	
convert_model lane|tl|obj|obj2 model_folder		
```