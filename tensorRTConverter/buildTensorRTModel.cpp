#include <algorithm>
#include "tensorNet.h"
#include <sstream>
#include <fstream>

using namespace nvinfer1;

void print_usage()
{
    printf("convert_model <lane|tl|obj|obj2> <model_folder>\n \
    make sure unetpp.caffemodel and unetpp.prototxt in lane folder\n \
    rfcn.caffemodel and rfcn.prototxt in tl folder\n \
    apollo_yolo.caffemodel and apollo_yolo.prototxt in obj folder\n\
    centernet.onnx in obj2 folder\n");
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        print_usage();
        return 0;
    }

    std::string model_type = argv[1];
    std::string model_folder = argv[2];
    std::string model, weight;
    std::vector<std::string> output_vector;
    if (model_type == "lane")
    {
        model = model_folder + "/unetpp.prototxt";
        weight = model_folder + "/unetpp.caffemodel" ;
        std::string OUTPUT0_BLOB_NAME = "Sigmoid_1";
        std::string INPUT_BLOB_NAME = "data";
        std::string OUTPUT1_BLOB_NAME = "out4";
        output_vector = {OUTPUT0_BLOB_NAME, OUTPUT1_BLOB_NAME};
    }
    else if(model_type=="tl")
    {
        model = model_folder + "/rfcn.prototxt";
        weight = model_folder + "/rfcn.caffemodel" ;
        std::string INPUT0_BLOB_NAME = "data";
        std::string INPUT1_BLOB_NAME = "im_info";
        std::string OUTPUT0_BLOB_NAME = "bbox_pred";
        std::string OUTPUT1_BLOB_NAME = "cls_perm_out";
        std::string OUTPUT2_BLOB_NAME = "rois";
        output_vector = {OUTPUT2_BLOB_NAME,OUTPUT1_BLOB_NAME,OUTPUT0_BLOB_NAME};
    }
    else if(model_type=="obj")
    {
        std::string INPUT_BLOB_NAME = "data";
        model = model_folder + "/apollo_yolo.prototxt";
        weight = model_folder + "/apollo_yolo.caffemodel" ;
        std::string OUTPUT1_BLOB_NAME = "loc_pred";
        std::string OUTPUT2_BLOB_NAME = "obj_perm";
        std::string OUTPUT3_BLOB_NAME = "cls_pred";
        std::string OUTPUT4_BLOB_NAME = "ori_pred";
        std::string OUTPUT5_BLOB_NAME = "dim_pred";
        std::string OUTPUT6_BLOB_NAME = "lof_pred";
        std::string OUTPUT7_BLOB_NAME = "lor_pred";
        output_vector = {OUTPUT1_BLOB_NAME,OUTPUT2_BLOB_NAME,OUTPUT3_BLOB_NAME,OUTPUT4_BLOB_NAME,OUTPUT5_BLOB_NAME,OUTPUT6_BLOB_NAME,OUTPUT7_BLOB_NAME};
    }
    else if(model_type=="obj2")
    {
        model = model_folder + "/unetpp.onnx";
        weight="";
    }
    else
    {
        printf("input error\n");
        print_usage();
        return 0;
    }
    
    //std::vector<std::string> output_vector = {OUTPUT0_BLOB_NAME};
    TensorNet tensorNet;
    static const uint32_t BATCH_SIZE = 1;
    tensorNet.LoadNetwork(model_folder,model,weight, output_vector, BATCH_SIZE);
    tensorNet.destroy();
    return 0;
}
