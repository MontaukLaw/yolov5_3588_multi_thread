
#include <opencv2/opencv.hpp>

#include "task/yolov5.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"

int main(int argc, char **argv)
{
    // model file path
    const char *model_file = argv[1];
    // input video
    const char *video_file = argv[2];
    // 参数：是否录像、绘制文字
    const bool record = argc > 3 ? atoi(argv[3]) : false;

    // 读取视频
    cv::VideoCapture cap(video_file);
    if (!cap.isOpened())
    {
        NN_LOG_ERROR("Failed to open video file: %s", video_file);
        return -1;
    }
    // 获取视频尺寸、帧率
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    NN_LOG_INFO("Video size: %d x %d, fps: %d", width, height, fps);

    // 初始化
    Yolov5 yolo;
    // 加载模型
    yolo.LoadModel(model_file);
    // 视频帧
    cv::Mat img;
    cv::VideoWriter writer;
    if (record)
    {
        // 写入视频mp4文件
        writer = cv::VideoWriter("result.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    }

    // all start
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (true)
    {
        // 开始计时
        auto start_1 = std::chrono::high_resolution_clock::now();

        // 读取视频帧
        cap >> img;
        if (img.empty())
        {
            NN_LOG_INFO("Video end.");
            break;
        }

        // 记录读取视频帧的时间：读取视频帧的时间
        auto end_1 = std::chrono::high_resolution_clock::now();
        // microseconds 微秒，milliseconds 毫秒，seconds 秒，1微妙=0.001毫秒 = 0.000001秒
        auto elapsed_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1).count() / 1000.0;

        // 开始计时
        auto start_2 = std::chrono::high_resolution_clock::now();
        // 检测结果
        std::vector<Detection> objects;
        
        // 运行模型
        yolo.Run(img, objects);
        // 绘制框，显示结果
        DrawDetections(img, objects);

        // 结束计时
        auto end_2 = std::chrono::high_resolution_clock::now();
        auto elapsed_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2).count() / 1000.0;

        // 算法1：计算单张图片的总耗时
        // 总时间
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_1).count() / 1000.0;
        // 计算帧率
        auto fps = 1000.0f / duration;

        // 如果计算帧率，输出帧率
        // 输出时间：读取视频帧的时间、模型运行时间、总时间
        NN_LOG_INFO("Method1 Time: %fms, %fms, %fms", elapsed_1, elapsed_2, duration);
        // 输出帧率：读取视频帧的帧率、模型运行帧率、总帧率
        NN_LOG_INFO("Method1 FPS: %f, %f, %f", 1000.0 / elapsed_1, 1000.0 / elapsed_2, fps);

        // 算法2：计算超过 1s 一共处理了多少张图片，即平均帧率
        frame_count++;
        auto elapsed_all_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_all).count() / 1000.f;
        // 每隔1秒打印一次
        if (elapsed_all_2 > 1000)
        {

            NN_LOG_INFO("Method2 Time:%fms, FPS:%f, Frame Count:%d", elapsed_all_2, frame_count / (elapsed_all_2 / 1000.0f), frame_count);
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }

        // 如果不计算帧率，就绘制总耗时和帧率, 写入视频帧。（因为method2会计入这个时间）
        if (record)
        {
            // 绘制总耗时和帧率
            auto time_str = std::to_string(duration) + "ms";
            auto fps_str = std::to_string(fps) + "fps";
            cv::putText(img, time_str, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            cv::putText(img, fps_str, cv::Point(50, 100), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);

            // 写入视频帧
            writer << img;
        }
    }

    return 0;
}