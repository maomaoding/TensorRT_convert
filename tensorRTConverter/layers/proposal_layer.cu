#include <iostream>
#include "layers/proposal_layer.h"
__device__ static int transform_box(float box[],
                                    const float dx, const float dy,
                                    const float d_log_w, const float d_log_h,
                                    const float img_W, const float img_H,
                                    const float min_box_W, const float min_box_H)
{
  // width & height of box
  const float w = box[2] - box[0] + (float)1;
  const float h = box[3] - box[1] + (float)1;
  // center location of box
  const float ctr_x = box[0] + (float)0.5 * w;
  const float ctr_y = box[1] + (float)0.5 * h;

  // new center location according to gradient (dx, dy)
  const float pred_ctr_x = dx * w + ctr_x;
  const float pred_ctr_y = dy * h + ctr_y;
  // new width & height according to gradient d(log w), d(log h)
  const float pred_w = exp(d_log_w) * w;
  const float pred_h = exp(d_log_h) * h;

  // update upper-left corner location
  box[0] = pred_ctr_x - (float)0.5 * pred_w;
  box[1] = pred_ctr_y - (float)0.5 * pred_h;
  // update lower-right corner location
  box[2] = pred_ctr_x + (float)0.5 * pred_w;
  box[3] = pred_ctr_y + (float)0.5 * pred_h;

  // adjust new corner locations to be within the image region,
  box[0] = max((float)0, min(box[0], img_W - (float)1));
  box[1] = max((float)0, min(box[1], img_H - (float)1));
  box[2] = max((float)0, min(box[2], img_W - (float)1));
  box[3] = max((float)0, min(box[3], img_H - (float)1));

  // recompute new width & height
  const float box_w = box[2] - box[0] + (float)1;
  const float box_h = box[3] - box[1] + (float)1;

  // check if new box's size >= threshold
  return (box_w >= min_box_W) * (box_h >= min_box_H);
}

__global__ static void enumerate_proposals_gpu(const int nthreads,
                                               const float bottom4d[],
                                               const float d_anchor4d[],
                                               const float anchors[],
                                               float proposals[],
                                               const int num_anchors,
                                               const int bottom_H, const int bottom_W,
                                               const float img_H, const float img_W,
                                               const float min_box_H, const float min_box_W,
                                               const int feat_stride)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    const int h = index / num_anchors / bottom_W;
    const int w = (index / num_anchors) % bottom_W;
    const int k = index % num_anchors;
    const float x = w * feat_stride;
    const float y = h * feat_stride;
    const float *p_box = d_anchor4d + h * bottom_W + w;
    const float *p_score = bottom4d + h * bottom_W + w;

    const int bottom_area = bottom_H * bottom_W;
    const float dx = p_box[(k * 4 + 0) * bottom_area];
    const float dy = p_box[(k * 4 + 1) * bottom_area];
    const float d_log_w = p_box[(k * 4 + 2) * bottom_area];
    const float d_log_h = p_box[(k * 4 + 3) * bottom_area];

    float *const p_proposal = proposals + index * 5;
    p_proposal[0] = x + anchors[k * 4 + 0];
    p_proposal[1] = y + anchors[k * 4 + 1];
    p_proposal[2] = x + anchors[k * 4 + 2];
    p_proposal[3] = y + anchors[k * 4 + 3];
    p_proposal[4] = transform_box(p_proposal,
                                  dx, dy, d_log_w, d_log_h,
                                  img_W, img_H, min_box_W, min_box_H) *
                    p_score[k * bottom_area];
  }
}
static void sort_box(float list_cpu[], const int start, const int end,
                     const int num_top)
{
  const float pivot_score = list_cpu[start * 5 + 4];
  int left = start + 1, right = end;
  float temp[5];
  while (left <= right)
  {
    while (left <= end && list_cpu[left * 5 + 4] >= pivot_score)
      ++left;
    while (right > start && list_cpu[right * 5 + 4] <= pivot_score)
      --right;
    if (left <= right)
    {
      for (int i = 0; i < 5; ++i)
      {
        temp[i] = list_cpu[left * 5 + i];
      }
      for (int i = 0; i < 5; ++i)
      {
        list_cpu[left * 5 + i] = list_cpu[right * 5 + i];
      }
      for (int i = 0; i < 5; ++i)
      {
        list_cpu[right * 5 + i] = temp[i];
      }
      ++left;
      --right;
    }
  }

  if (right > start)
  {
    for (int i = 0; i < 5; ++i)
    {
      temp[i] = list_cpu[start * 5 + i];
    }
    for (int i = 0; i < 5; ++i)
    {
      list_cpu[start * 5 + i] = list_cpu[right * 5 + i];
    }
    for (int i = 0; i < 5; ++i)
    {
      list_cpu[right * 5 + i] = temp[i];
    }
  }

  if (start < right - 1)
  {
    sort_box(list_cpu, start, right - 1, num_top);
  }
  if (right + 1 < num_top && right + 1 < end)
  {
    sort_box(list_cpu, right + 1, end, num_top);
  }
}
__global__ static void retrieve_rois_gpu(const int nthreads,
                                         const int item_index,
                                         const float proposals[],
                                         const int roi_indices[],
                                         float rois[],
                                         float roi_scores[])
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    const float *const proposals_index = proposals + roi_indices[index] * 5;
    rois[index * 5 + 0] = item_index;
    rois[index * 5 + 1] = proposals_index[0];
    rois[index * 5 + 2] = proposals_index[1];
    rois[index * 5 + 3] = proposals_index[2];
    rois[index * 5 + 4] = proposals_index[3];
  }
}
void ProposalLayer::forward_gpu(const void *const *inputs)
{
  const float *p_bottom_item = reinterpret_cast<const float *>(inputs[0]);
  const float *p_d_anchor_item = reinterpret_cast<const float *>(inputs[1]);
  //const float* p_img_info_cpu = reinterpret_cast<const float*>(inputs[2]);
  float *p_img_info_cpu = new float[3];
  cudaMemcpyAsync(p_img_info_cpu, reinterpret_cast<const float *>(inputs[2]), 3 * sizeof(float), cudaMemcpyDeviceToHost);

  vector<int> proposals_shape(2);
  vector<int> top_shape(2);
  proposals_shape[0] = 0;
  proposals_shape[1] = 5;

  // bottom shape: (2 x num_anchors) x H x W
  const int bottom_H = input_cls_shape_[1];
  const int bottom_W = input_cls_shape_[2];
  // input image height & width
  const float img_H = p_img_info_cpu[0];
  const float img_W = p_img_info_cpu[1];
  // scale factor for height & width
  const float scale_H = p_img_info_cpu[2];
  const float scale_W = p_img_info_cpu[3];
  // minimum box width & height
  const float min_box_H = min_size_ * scale_H;
  const float min_box_W = min_size_ * scale_W;
  // number of all proposals = num_anchors * H * W
  const int num_proposals = anchor_shape_[0] * bottom_H * bottom_W;
  // number of top-n proposals before NMS
  const int pre_nms_topn = std::min(num_proposals, pre_nms_topn_);
  // number of final RoIs
  int num_rois = 0;

  // enumerate all proposals
  //   num_proposals = num_anchors * H * W
  //   (x1, y1, x2, y2, score) for each proposal
  // NOTE: for bottom, only foreground scores are passed
  proposals_shape[0] = num_proposals;
  cudaMallocManaged(&proposals_, proposals_shape[0] * proposals_shape[1] * sizeof(float));
  cudaMemset(proposals_, 0, proposals_shape[0] * proposals_shape[1] * sizeof(float));

  enumerate_proposals_gpu<<<TENSORRT_GET_BLOCKS(num_proposals),
                            TENSORRT_CUDA_NUM_THREADS>>>(
      num_proposals,
      p_bottom_item + num_proposals, p_d_anchor_item,
      anchors_, proposals_, anchor_shape_[0],
      bottom_H, bottom_W, img_H, img_W, min_box_H, min_box_W,
      feat_stride_);

  //cudaDeviceSynchronize();
  float *proposals_cpu = new float[proposals_shape[0] * proposals_shape[1]];
  cudaMemcpy(proposals_cpu, proposals_, proposals_shape[0] * proposals_shape[1] * sizeof(float), cudaMemcpyDeviceToHost);
  sort_box(proposals_cpu, 0, num_proposals - 1, pre_nms_topn_);
  cudaMemcpy(proposals_, proposals_cpu, proposals_shape[0] * proposals_shape[1] * sizeof(float), cudaMemcpyHostToDevice);

  int *roi_indices_cpu = new int[post_nms_topn_];
  cudaMemcpy(roi_indices_cpu, roi_indices_, post_nms_topn_ * sizeof(int), cudaMemcpyDeviceToHost);
  nms_gpu(pre_nms_topn, proposals_, nms_mask_,
          roi_indices_cpu, &num_rois,
          0, nms_thresh_, post_nms_topn_);
  cudaMemcpy(roi_indices_, roi_indices_cpu, post_nms_topn_ * sizeof(int), cudaMemcpyHostToDevice);

  retrieve_rois_gpu<<<TENSORRT_GET_BLOCKS(num_rois),
                      TENSORRT_CUDA_NUM_THREADS>>>(
      num_rois, 0, proposals_, roi_indices_,
      rois_, nullptr);
  delete roi_indices_cpu;
  roi_indices_cpu = nullptr;
  delete proposals_cpu;
  proposals_cpu = nullptr;
  cudaFree(proposals_);
  delete p_img_info_cpu;
  p_img_info_cpu = nullptr;
}
