#include "utils.h"
//————————————————————————————————————————————————————————————————————————————————————————
//proposal layer fucs
__device__ static float iou(const float A[], const float B[])
{
  // overlapped region (= box)
  const float x1 = max(A[0], B[0]);
  const float y1 = max(A[1], B[1]);
  const float x2 = min(A[2], B[2]);
  const float y2 = min(A[3], B[3]);

  // intersection area
  const float width = max((float)0, x2 - x1 + (float)1);
  const float height = max((float)0, y2 - y1 + (float)1);
  const float area = width * height;

  // area of A, B
  const float A_area = (A[2] - A[0] + (float)1) * (A[3] - A[1] + (float)1);
  const float B_area = (B[2] - B[0] + (float)1) * (B[3] - B[1] + (float)1);

  // IoU
  return area / (A_area + B_area - area);
}

static const int nms_block_size = 64;
// given box proposals, compute overlap between all box pairs
// (overlap = intersection area / union area)
// and then set mask-bit to 1 if a pair is significantly overlapped
//   num_boxes: number of box proposals given
//   boxes: "num_boxes x 5" array (x1, y1, x2, y2, score)
//   nms_thresh: threshold for determining "significant overlap"
//               if "intersection area / union area > nms_thresh",
//               two boxes are thought of as significantly overlapped
// the all-pair computation (num_boxes x num_boxes) is done by
// divide-and-conquer:
//   each GPU block (bj, bi) computes for "64 x 64" box pairs (j, i),
//     j = bj * 64 + { 0, 1, ..., 63 }
//     i = bi * 64 + { 0, 1, ..., 63 },
//   and each "1 x 64" results is saved into a 64-bit mask
//     mask: "num_boxes x num_blocks" array
//     for mask[j][bi], "di-th bit = 1" means:
//       box j is significantly overlapped with box i,
//       where i = bi * 64 + di
__global__ static void nms_mask(const float boxes[], unsigned long long mask[],
                                const int num_boxes, const float nms_thresh)
{
  // block region
  //   j = j_start + { 0, ..., dj_end - 1 }
  //   i = i_start + { 0, ..., di_end - 1 }
  const int i_start = blockIdx.x * nms_block_size;
  const int di_end = min(num_boxes - i_start, nms_block_size);
  const int j_start = blockIdx.y * nms_block_size;
  const int dj_end = min(num_boxes - j_start, nms_block_size);

  // copy all i-th boxes to GPU cache
  //   i = i_start + { 0, ..., di_end - 1 }
  __shared__ float boxes_i[nms_block_size * 4];
  {
    const int di = threadIdx.x;
    if (di < di_end)
    {
      boxes_i[di * 4 + 0] = boxes[(i_start + di) * 5 + 0];
      boxes_i[di * 4 + 1] = boxes[(i_start + di) * 5 + 1];
      boxes_i[di * 4 + 2] = boxes[(i_start + di) * 5 + 2];
      boxes_i[di * 4 + 3] = boxes[(i_start + di) * 5 + 3];
    }
  }
  __syncthreads();

  // given j = j_start + dj,
  //   check whether box i is significantly overlapped with box j
  //   (i.e., IoU(box j, box i) > threshold)
  //   for all i = i_start + { 0, ..., di_end - 1 } except for i == j
  {
    const int dj = threadIdx.x;
    if (dj < dj_end)
    {
      // box j
      const float *const box_j = boxes + (j_start + dj) * 5;

      // mask for significant overlap
      //   if IoU(box j, box i) > threshold,  di-th bit = 1
      unsigned long long mask_j = 0;

      // check for all i = i_start + { 0, ..., di_end - 1 }
      // except for i == j
      const int di_start = (i_start == j_start) ? (dj + 1) : 0;
      for (int di = di_start; di < di_end; ++di)
      {
        // box i
        const float *const box_i = boxes_i + di * 4;

        // if IoU(box j, box i) > threshold,  di-th bit = 1
        if (iou(box_j, box_i) > nms_thresh)
        {
          mask_j |= 1ULL << di;
        }
      }

      // mask: "num_boxes x num_blocks" array
      //   for mask[j][bi], "di-th bit = 1" means:
      //     box j is significantly overlapped with box i = i_start + di,
      //     where i_start = bi * block_size
      {
        const int num_blocks = DIV_THEN_CEIL(num_boxes, nms_block_size);
        const int bi = blockIdx.x;
        mask[(j_start + dj) * num_blocks + bi] = mask_j;
      }
    } // endif dj < dj_end
  }
}

void nms_gpu(const int num_boxes,
             const float boxes_gpu[],
             int *p_mask,
             int index_out_cpu[],
             int *const num_out,
             const int base_index,
             const float nms_thresh, const int max_num_out)
{
  const int num_blocks = DIV_THEN_CEIL(num_boxes, nms_block_size);
  const dim3 blocks(num_blocks, num_blocks);
  std::vector<int> mask_shape(2);
  mask_shape[0] = num_boxes;
  mask_shape[1] = num_blocks * sizeof(unsigned long long) / sizeof(int);
  cudaMallocManaged(&p_mask, mask_shape[0] * mask_shape[1] * sizeof(int));
  // find all significantly-overlapped pairs of boxes
  nms_mask<<<blocks, nms_block_size>>>(
      boxes_gpu, (unsigned long long *)p_mask,
      num_boxes, nms_thresh);
  //cudaDeviceSynchronize();
  int *p_mask_cpu = new int[mask_shape[0] * mask_shape[1]];
  cudaMemcpy(p_mask_cpu, p_mask, mask_shape[0] * mask_shape[1] * sizeof(int), cudaMemcpyDeviceToHost);
  // discard i-th box if it is significantly overlapped with
  // one or more previous (= scored higher) boxes
  int num_selected = 0;
  std::vector<unsigned long long> dead_bit(num_blocks);
  for (int i = 0; i < num_blocks; ++i)
  {
    dead_bit[i] = 0;
  }
  for (int i = 0; i < num_boxes; ++i)
  {
    const int nblock = i / nms_block_size;
    const int inblock = i % nms_block_size;

    if (!(dead_bit[nblock] & (1ULL << inblock)))
    {
      index_out_cpu[num_selected++] = base_index + i;
      const unsigned long long *const mask_i = (unsigned long long *)p_mask_cpu + i * num_blocks;
      for (int j = nblock; j < num_blocks; ++j)
      {
        dead_bit[j] |= mask_i[j];
      }

      if (num_selected == max_num_out)
      {
        break;
      }
    }
  }
  *num_out = num_selected;
  delete p_mask_cpu;
  p_mask_cpu = nullptr;
  cudaFree(p_mask);
}
//————————————————————————————————————————————————————————————————————————————————————————
