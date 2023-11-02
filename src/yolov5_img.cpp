
#include <opencv2/opencv.hpp>

#include "task/yolov5.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"

int main(int argc, char **argv)
{
    // model file path
    const char *model_file = argv[1];
    // input img path
    const char *img_file = argv[2];
    // 读取图片
    cv::Mat img = cv::imread(img_file);
    // print img size
    NN_LOG_INFO("img size: %d x %d", img.cols, img.rows);

    // 初始化
    Yolov5 yolo;
    // 加载模型
    yolo.LoadModel(model_file);

    // 运行模型
    std::vector<Detection> objects;
    yolo.Run(img, objects);

    // 显示结果
    DrawDetections(img, objects);

    // 保存结果
    cv::imwrite("result.jpg", img);

    return 0;
}