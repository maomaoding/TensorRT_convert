#include "layers/proposal_layer.h"

#define ROUND(x) ((int)((x) + (float)0.5))
ProposalLayer::ProposalLayer(void **output, vector<int> input_cls_shape, vector<int> input_bbox_shape, vector<int> output_shape)
{
  rois_ = reinterpret_cast<float *>(output[0]);
  int size_output = output_shape[0] * output_shape[1] * output_shape[2];
  cudaMemset(rois_, 0, size_output * sizeof(float));
  rois_host_ = new float[size_output];
  cudaMemcpyAsync(rois_host_, rois_, size_output * sizeof(float), cudaMemcpyDeviceToHost);
  output_shape_ = output_shape;
  input_cls_shape_ = input_cls_shape;
  input_bbox_shape_ = input_bbox_shape;
  LayerSetUp();
}

ProposalLayer::~ProposalLayer()
{
  delete rois_host_;
  rois_host_ = nullptr;
  cudaFree(anchors_);
  cudaFree(roi_indices_);
}

static int transform_box(float box[],
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
  box[0] = std::max((float)0, std::min(box[0], img_W - (float)1));
  box[1] = std::max((float)0, std::min(box[1], img_H - (float)1));
  box[2] = std::max((float)0, std::min(box[2], img_W - (float)1));
  box[3] = std::max((float)0, std::min(box[3], img_H - (float)1));

  // recompute new width & height
  const float box_w = box[2] - box[0] + (float)1;
  const float box_h = box[3] - box[1] + (float)1;

  // check if new box's size >= threshold
  return (box_w >= min_box_W) * (box_h >= min_box_H);
}
static void enumerate_proposals_cpu(const float bottom4d[],
                                    const float d_anchor4d[],
                                    const float anchors[],
                                    float proposals[],
                                    const int num_anchors,
                                    const int bottom_H, const int bottom_W,
                                    const float img_H, const float img_W,
                                    const float min_box_H, const float min_box_W,
                                    const int feat_stride)
{
  float *p_proposal = proposals;
  const int bottom_area = bottom_H * bottom_W;

  for (int h = 0; h < bottom_H; ++h)
  {
    for (int w = 0; w < bottom_W; ++w)
    {
      const float x = w * feat_stride;
      const float y = h * feat_stride;
      const float *p_box = d_anchor4d + h * bottom_W + w;
      const float *p_score = bottom4d + h * bottom_W + w;
      for (int k = 0; k < num_anchors; ++k)
      {
        const float dx = p_box[(k * 4 + 0) * bottom_area];
        const float dy = p_box[(k * 4 + 1) * bottom_area];
        const float d_log_w = p_box[(k * 4 + 2) * bottom_area];
        const float d_log_h = p_box[(k * 4 + 3) * bottom_area];

        p_proposal[0] = x + anchors[k * 4 + 0];
        p_proposal[1] = y + anchors[k * 4 + 1];
        p_proposal[2] = x + anchors[k * 4 + 2];
        p_proposal[3] = y + anchors[k * 4 + 3];
        p_proposal[4] = transform_box(p_proposal,
                                      dx, dy, d_log_w, d_log_h,
                                      img_W, img_H, min_box_W, min_box_H) *
                        p_score[k * bottom_area];
        p_proposal += 5;
      } // endfor k
    }   // endfor w
  }     // endfor h
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
static void retrieve_rois_cpu(const int num_rois,
                              const int item_index,
                              const float proposals[],
                              const int roi_indices[],
                              float rois[],
                              float roi_scores[])
{
  for (int i = 0; i < num_rois - 1; ++i)
  {
    const float *const proposals_index = proposals + roi_indices[i] * 5;
    rois[i * 5 + 0] = item_index;
    rois[i * 5 + 1] = proposals_index[0];
    rois[i * 5 + 2] = proposals_index[1];
    rois[i * 5 + 3] = proposals_index[2];
    rois[i * 5 + 4] = proposals_index[3];
    if (roi_scores)
    {
      roi_scores[i] = proposals_index[4];
    }
  }
}
void ProposalLayer::forward_cpu(const void *const *inputs)
{

  int size_cls = input_cls_shape_[0] * input_cls_shape_[1] * input_cls_shape_[2];
  float *input_cls = new float[size_cls];
  cudaMemcpyAsync(input_cls, reinterpret_cast<const float *>(inputs[0]), size_cls * sizeof(float), cudaMemcpyDeviceToHost);

  int size_bbox = input_bbox_shape_[0] * input_bbox_shape_[1] * input_bbox_shape_[2];
  float *input_bbox = new float[size_bbox];
  cudaMemcpyAsync(input_bbox, reinterpret_cast<const float *>(inputs[1]), size_bbox * sizeof(float), cudaMemcpyDeviceToHost);

  float *input_im_info = new float[3];
  cudaMemcpyAsync(input_im_info, reinterpret_cast<const float *>(inputs[2]), 3 * sizeof(float), cudaMemcpyDeviceToHost);

  vector<int> proposals_shape(2);
  vector<int> top_shape(2);
  proposals_shape[0] = 0;
  proposals_shape[1] = 5;

  // bottom shape: (2 x num_anchors) x H x W
  const int bottom_H = input_cls_shape_[1];
  const int bottom_W = input_cls_shape_[2];
  // input image height & width
  const float img_H = input_im_info[0];
  const float img_W = input_im_info[1];
  // scale factor for height & width
  const float scale_H = input_im_info[2];
  const float scale_W = input_im_info[2];
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
  proposals_ = new float[proposals_shape[0] * proposals_shape[1]];

  enumerate_proposals_cpu(
      input_cls, input_bbox,
      anchors_, proposals_, anchor_shape_[0],
      bottom_H, bottom_W, img_H, img_W, min_box_H, min_box_W,
      feat_stride_);

  sort_box(proposals_, 0, num_proposals - 1, pre_nms_topn_);

  nms_cpu(pre_nms_topn, proposals_,
          roi_indices_, &num_rois,
          0, nms_thresh_, post_nms_topn_);

  retrieve_rois_cpu(
      num_rois, 0, proposals_, roi_indices_,
      rois_host_, nullptr);
  //std::cout<<num_rois<<std::endl;
  int size_output = output_shape_[0] * output_shape_[1] * output_shape_[2];
  //rois_=new float[size_output];
  cudaMemcpyAsync(rois_, rois_host_, size_output * sizeof(float), cudaMemcpyHostToDevice);
  delete proposals_;
  proposals_ = nullptr;
  delete input_cls;
  input_cls = nullptr;

  delete input_bbox;
  input_bbox = nullptr;

  delete input_im_info;
  input_im_info = nullptr;
}

static void generate_anchors(int base_size,
                             const float ratios[],
                             const float scales[],
                             const int num_ratios,
                             const int num_scales,
                             float anchors[])
{
  // base box's width & height & center location
  const float base_area = (float)(base_size * base_size);
  const float center = (float)0.5 * (base_size - (float)1);

  // enumerate all transformed boxes
  float *p_anchors = anchors;
  for (int i = 0; i < num_ratios; ++i)
  {
    // transformed width & height for given ratio factors
    const float ratio_w = (float)ROUND(sqrt(base_area / ratios[i]));
    const float ratio_h = (float)ROUND(ratio_w * ratios[i]);

    for (int j = 0; j < num_scales; ++j)
    {
      // transformed width & height for given scale factors
      const float scale_w = (float)0.5 * (ratio_w * scales[j] - (float)1);
      const float scale_h = (float)0.5 * (ratio_h * scales[j] - (float)1);

      // (x1, y1, x2, y2) for transformed box
      p_anchors[0] = center - scale_w;
      p_anchors[1] = center - scale_h;
      p_anchors[2] = center + scale_w;
      p_anchors[3] = center + scale_h;
      p_anchors += 4;
    } // endfor jfloat
  }
}

void ProposalLayer::LayerSetUp()
{
  base_size_ = BASE_SIZE;
  feat_stride_ = FEAT_STRIDE;
  pre_nms_topn_ = RPN_PRE_NUM;
  post_nms_topn_ = RPN_NMS_MAX;
  nms_thresh_ = RPN_NMS_THRESHOLD;
  min_size_ = 8;
  vector<float> ratios{0.5, 1, 2};
  vector<float> scales{8, 16, 32};
  vector<int> anchors_shape(2);

  anchor_shape_.clear();
  anchors_shape[0] = ratios.size() * scales.size();
  anchors_shape[1] = 4;
  anchor_shape_ = anchors_shape;
  cudaMallocManaged(&anchors_, anchors_shape[0] * anchors_shape[1] * sizeof(float));
  float *anchors_cpu = new float[anchors_shape[0] * anchors_shape[1]];
  //cudaMemcpy(anchors_cpu,anchors_, anchors_shape[0]*anchors_shape[1]*sizeof(float),cudaMemcpyDeviceToHost);
  //cudaDeviceSynchronize();
  generate_anchors(base_size_, &ratios[0], &scales[0],
                   ratios.size(), scales.size(),
                   anchors_cpu);
  cudaMemcpy(anchors_, anchors_cpu, anchors_shape[0] * anchors_shape[1] * sizeof(float), cudaMemcpyHostToDevice);
  delete anchors_cpu;
  anchors_cpu = nullptr;
  //roi_indices_=new int[post_nms_topn_];
  cudaMallocManaged(&roi_indices_, post_nms_topn_ * sizeof(int));
}
