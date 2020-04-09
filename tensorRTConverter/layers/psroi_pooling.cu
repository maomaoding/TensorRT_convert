#include "layers/psroi_pooling.h"
#define ROUND(x) ((int)((x) + (float)0.5))
using std::max;
using std::min;
PsroiPooling::PsroiPooling(void **output, vector<int> input_feature_shape, vector<int> input_roi_shape, vector<int> output_shape, float spatial_sacle)
{
    input_featrue_shape_ = input_feature_shape;
    input_roi_shape_ = input_roi_shape;
    output_shape_ = output_shape;
    spatial_scale_ = spatial_sacle;
    top_data_ = reinterpret_cast<float *>(output[0]);
}

__global__ void PSROIPoolingForward(
    const int nthreads,
    const float *bottom_data,
    const float spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const float *bottom_rois,
    const int output_dim,
    const int group_size,
    float *top_data,
    int *mapping_channel)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
        float roi_start_w =
            static_cast<float>(ROUND(bottom_rois[1])) * spatial_scale;
        float roi_start_h =
            static_cast<float>(ROUND(bottom_rois[2])) * spatial_scale;
        float roi_end_w =
            static_cast<float>(ROUND(bottom_rois[3]) + 1.) * spatial_scale;
        float roi_end_h =
            static_cast<float>(ROUND(bottom_rois[4]) + 1.) * spatial_scale;
        //printf("%f %f %f %f\n",roi_start_w,roi_start_h,roi_end_w,roi_end_h);
        // Force too small ROIs to be 1x1
        float roi_width = max(roi_end_w - roi_start_w, 0.1); // avoid 0
        float roi_height = max(roi_end_h - roi_start_h, 0.1);

        // Compute w and h at bottom
        float bin_size_h = roi_height / static_cast<float>(pooled_height);
        float bin_size_w = roi_width / static_cast<float>(pooled_width);

        int hstart = floor(static_cast<float>(ph) * bin_size_h + roi_start_h);
        int wstart = floor(static_cast<float>(pw) * bin_size_w + roi_start_w);
        int hend = ceil(static_cast<float>(ph + 1) * bin_size_h + roi_start_h);
        int wend = ceil(static_cast<float>(pw + 1) * bin_size_w + roi_start_w);
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        int gw = pw;
        int gh = ph;
        int c = (ctop * group_size + gh) * group_size + gw;

        bottom_data += (roi_batch_ind * channels + c) * height * width;
        float out_sum = 0;
        for (int h = hstart; h < hend; ++h)
        {
            for (int w = wstart; w < wend; ++w)
            {
                int bottom_index = h * width + w;
                out_sum += bottom_data[bottom_index];
            }
        }

        float bin_area = (hend - hstart) * (wend - wstart);
        top_data[index] = is_empty ? 0. : out_sum / bin_area;
        mapping_channel[index] = c;
    }
}
void PsroiPooling::forward_gpu(const void *const *inputs)
{
    const float *bottom_data = reinterpret_cast<const float *>(inputs[0]);
    const float *bottom_rois = reinterpret_cast<const float *>(inputs[1]);
    int *mapping_channel_ptr = nullptr;
    //std::cout<<input_roi_shape_[0]<<std::endl;
    cudaMalloc((void **)&mapping_channel_ptr, input_roi_shape_[0] * output_shape_[0] * output_shape_[1] * output_shape_[2] * sizeof(int));

    cudaMemset(mapping_channel_ptr, -1, input_roi_shape_[0] * output_shape_[0] * output_shape_[1] * output_shape_[2] * sizeof(int));
    cudaMemset(top_data_, 0, input_roi_shape_[0] * output_shape_[0] * output_shape_[1] * output_shape_[2] * sizeof(float));
    int count = input_roi_shape_[0] * output_shape_[0] * output_shape_[1] * output_shape_[2];
    PSROIPoolingForward<<<TENSORRT_GET_BLOCKS(count),
                          TENSORRT_CUDA_NUM_THREADS>>>(count, bottom_data, spatial_scale_,
                                                       input_featrue_shape_[0], input_featrue_shape_[1], input_featrue_shape_[2], output_shape_[1],
                                                       output_shape_[2], bottom_rois, output_shape_[0], output_shape_[1],
                                                       top_data_, mapping_channel_ptr);
    cudaFree(mapping_channel_ptr);
}