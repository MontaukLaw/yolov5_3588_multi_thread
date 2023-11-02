
#include <opencv2/opencv.hpp>

#include "task/yolov5.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"

#include "task/yolov5_thread_pool.h"

static int g_frame_start_id = 0; // 读取视频帧的索引
static int g_frame_end_id = 0;   // 模型处理完的索引

// 创建线程池
static Yolov5ThreadPool *g_pool = nullptr;
bool end = false;

void get_results(int width = 1280, int height = 720, int fps = 30)
{
    // int64_t frame_id = 0;
    // 记录开始时间
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    // cv::VideoWriter writer;
    // writer = cv::VideoWriter("result_pool.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

    while (true)
    {

        // 结果
        cv::Mat img;
        auto ret = g_pool->getTargetImgResult(img, g_frame_end_id++);
        // 如果读取完毕，且模型处理完毕，结束
        if (end && ret != NN_SUCCESS)
        {
            g_pool->stopAll();
            break;
        }
        
        // printf("g_frame_end_id: %d\n", g_frame_end_id);
        // printf("g_frame_start_id: %d\n", g_frame_start_id);
        // save img
        // cv::imwrite("output/" + std::to_string(g_frame_end_id) + ".jpg", img);
        // 写入视频帧
        // writer << img;

        // 算法2：计算超过 1s 一共处理了多少张图片
        frame_count++;
        // all end
        auto end_all = std::chrono::high_resolution_clock::now();
        auto elapsed_all_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count() / 1000.f;
        // 每隔1秒打印一次
        if (elapsed_all_2 > 1000)
        {
            NN_LOG_INFO("Method2 Time:%fms, FPS:%f, Frame Count:%d", elapsed_all_2, frame_count / (elapsed_all_2 / 1000.0f), frame_count);
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }
    // 结束所有线程
    g_pool->stopAll();
    NN_LOG_INFO("Get results end.");
}
// 读取视频帧，提交任务
void read_stream(const char *video_file)
{
    // 读取视频
    cv::VideoCapture cap(video_file);
    if (!cap.isOpened())
    {
        NN_LOG_ERROR("Failed to open video file: %s", video_file);
    }
    // 获取视频尺寸、帧率
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    NN_LOG_INFO("Video size: %d x %d, fps: %d", width, height, fps);

    // 画面
    cv::Mat img;

    while (true)
    {

        // 读取视频帧
        cap >> img;
        if (img.empty())
        {
            NN_LOG_INFO("Video end.");
            // 结束所有线程
            // sleep 5s
            // std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            end = true;
            break;
        }

        // 提交任务，这里使用clone，因为不这样数据在内存中可能不连续，导致绘制错误
        g_pool->submitTask(img.clone(), g_frame_start_id++);
    }
    // 释放资源
    cap.release();
}

int main(int argc, char **argv)
{
    // model file path
    std::string model_file = argv[1];
    // input video
    const char *video_file = argv[2];
    // 参数：线程池数量
    const int num_threads = (argc > 3) ? atoi(argv[3]) : 12;

    // 线程1：读取视频帧，提交任务
    // 线程池：模型运行
    // 线程2：拿到结果，绘制结果

    // 实例化线程池
    g_pool = new Yolov5ThreadPool();
    g_pool->setUp(model_file, num_threads);

    // 读取视频
    std::thread read_stream_thread(read_stream, video_file);
    // 启动结果线程
    std::thread result_thread(get_results, 1280, 720, 25);

    // 等待线程结束
    read_stream_thread.join();
    result_thread.join();

    return 0;
}