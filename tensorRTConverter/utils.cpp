#include "utils.h"
#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <vector>
#include <dirent.h>
using std::max;
using std::min;
//——————————————————————————————————————————————————————————————————————————————
// 作用：获取文件夹中所有文件的路径
// 参数1：输入 文件夹路径
// 参数2：输出 文件夹中所有文件路径数组
//——————————————————————————————————————————————————————————————————————————————
void getFiles(const char *path, std::vector<std::string> &files)
{
  const std::string path0 = path;
  DIR *pDir;
  struct dirent *ptr;

  struct stat s;
  lstat(path, &s);

  if (!S_ISDIR(s.st_mode))
  {
    return;
  }

  if (!(pDir = opendir(path)))
  {
    return;
  }
  int i = 0;
  std::string subFile;
  while ((ptr = readdir(pDir)) != 0)
  {
    subFile = ptr->d_name;
    if (subFile == "." || subFile == "..")
      continue;
    subFile = path0 + subFile;
    files.push_back(subFile);
  }
  closedir(pDir);
}

//——————————————————————————————————————————————————————————————————————————————
// 作用：字符串分割，等同于numpy的spilt
// 参数1：需要分割的字符串
// 参数2：分割所用字符
// 返回值：std::vector<std::string>的数组，长度为分割出来的字符数
//——————————————————————————————————————————————————————————————————————————————
std::vector<std::string> supersplit(const std::string &s, const std::string &c)
{
  std::vector<std::string> v;
  std::string::size_type pos1, pos2;
  size_t len = s.length();
  pos2 = s.find(c);
  pos1 = 0;
  while (std::string::npos != pos2)
  {
    v.emplace_back(s.substr(pos1, pos2 - pos1));

    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if (pos1 != len)
    v.emplace_back(s.substr(pos1));
  return v;
}
//——————————————————————————————————————————————————————————————————————————————
// 作用：获取输入模型所在文件夹的路径
// 参数1：input:模型的路径（/home/test/test.pth）
// 返回值：模型所在的文件夹的路径/home/test/）
//——————————————————————————————————————————————————————————————————————————————
std::string getModelDir(std::string input)
{
  std::vector<std::string> split = supersplit(input, "/");
  std::string out = "";
  for (int i = 0; i < split.size() - 1; i++)
  {
    out += split[i] + "/";
  }
  return out;
}
//————————————————————————————————————————————————————————————————————————————————————————
//proposal layer fucs
static float iou(const float A[], const float B[])
{
  if (A[0] > B[2] || A[1] > B[3] || A[2] < B[0] || A[3] < B[1])
  {
    return 0;
  }

  // overlapped region (= box)
  const float x1 = std::max(A[0], B[0]);
  const float y1 = std::max(A[1], B[1]);
  const float x2 = std::min(A[2], B[2]);
  const float y2 = std::min(A[3], B[3]);

  // intersection area
  const float width = std::max((float)0, x2 - x1 + (float)1);
  const float height = std::max((float)0, y2 - y1 + (float)1);
  const float area = width * height;

  // area of A, B
  const float A_area = (A[2] - A[0] + (float)1) * (A[3] - A[1] + (float)1);
  const float B_area = (B[2] - B[0] + (float)1) * (B[3] - B[1] + (float)1);

  // IoU
  return area / (A_area + B_area - area);
}
void nms_cpu(const int num_boxes,
             const float boxes[],
             int index_out[],
             int *const num_out,
             const int base_index,
             const float nms_thresh, const int max_num_out)
{
  int count = 0;
  std::vector<char> is_dead(num_boxes);
  for (int i = 0; i < num_boxes; ++i)
  {
    is_dead[i] = 0;
  }

  for (int i = 0; i < num_boxes; ++i)
  {
    if (is_dead[i])
    {
      continue;
    }

    index_out[count++] = base_index + i;
    if (count == max_num_out)
    {
      break;
    }

    for (int j = i + 1; j < num_boxes; ++j)
    {
      if (!is_dead[j] && iou(&boxes[i * 5], &boxes[j * 5]) > nms_thresh)
      {
        is_dead[j] = 1;
      }
    }
  }

  *num_out = count;
  is_dead.clear();
}
//————————————————————————————————————————————————————————————————————————————————————————