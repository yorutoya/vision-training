#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;

// 1. 参数与数据结构
constexpr int INPUT_SIZE = 640;
constexpr int PREDICTION_COUNT = 25200;
constexpr int PREDICTION_WIDTH = 22;
constexpr float CONFIDENCE_THRESHOLD = 0.5f;
constexpr float NMS_THRESHOLD = 0.5f;
constexpr double VIDEO_FPS = 30.0;

const std::array<std::string, 4> COLOR_NAMES = {
    "blue", "red", "neutral", "purple"
};
const std::array<std::string, 9> PATTERN_NAMES = {
    "sentry", "1", "2", "3", "4", "5", "outpost", "base_small", "base_big"
};

struct Arguments {
    std::filesystem::path model;
    std::string video;
    std::string calibration;
    std::filesystem::path output;
    std::filesystem::path preview;
};

struct Calibration {
    cv::Mat camera;
    cv::Mat distortion;
    cv::Size imageSize;
};

struct ModelInput {
    std::vector<Ort::Float16_t> pixels;
    double scaleX;
    double scaleY;
    int padLeft;
    int padTop;
};

struct Detection {
    // 顺序固定为：左上、左下、右下、右上。
    std::vector<cv::Point2d> corners;
    cv::Rect2d box;
    float confidence;
    int color;
    int pattern;
};

Arguments readArguments(int argc, char* argv[]) {
    Arguments args;

    // 兼容之前 CLion 中填写的三个位置参数。
    if (argc == 4 && std::string(argv[1]).rfind("--", 0) != 0) {
        args.model = argv[1];
        args.video = argv[2];
        args.calibration = argv[3];
        std::string stem = std::filesystem::path(args.video).stem().string();
        args.output = std::filesystem::path("results") / (stem + ".jsonl");
        args.preview = std::filesystem::path("results") / (stem + "_preview.png");
        return args;
    }

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing argument value");
        }
        std::string key = argv[i];
        if (key == "--model") {
            args.model = argv[i + 1];
        } else if (key == "--video") {
            args.video = argv[i + 1];
        } else if (key == "--calibration") {
            args.calibration = argv[i + 1];
        } else if (key == "--output") {
            args.output = argv[i + 1];
        } else if (key == "--preview") {
            args.preview = argv[i + 1];
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }

    if (args.model.empty() || args.video.empty() ||
        args.calibration.empty() || args.output.empty()) {
        throw std::runtime_error(
            "Usage: vision_detection --model PATH --video PATH "
            "--calibration PATH --output PATH [--preview PATH]");
    }
    return args;
}

Calibration loadCalibration(const std::string& path) {
    YAML::Node data = YAML::LoadFile(path);
    Calibration result;
    result.imageSize = cv::Size(data["image_width"].as<int>(),
                               data["image_height"].as<int>());
    result.camera = cv::Mat(3, 3, CV_64F);
    result.distortion = cv::Mat(1, 5, CV_64F);

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            result.camera.at<double>(row, col) =
                data["camera_matrix"][row][col].as<double>();
        }
    }
    for (int i = 0; i < 5; ++i) {
        result.distortion.at<double>(0, i) = data["distort_coeffs"][i].as<double>();
    }
    if (!cv::checkRange(result.camera) || !cv::checkRange(result.distortion) ||
        result.camera.at<double>(0, 0) <= 0 || result.camera.at<double>(1, 1) <= 0 ||
        result.imageSize.width <= 0 || result.imageSize.height <= 0) {
        throw std::runtime_error("Invalid camera calibration");
    }
    return result;
}

// 2. 预处理：等比例缩放、补边、RGB、CHW、float16。
ModelInput preprocess(const cv::Mat& frame) {
    ModelInput result;
    double scale = std::min(double(INPUT_SIZE) / frame.cols,
                            double(INPUT_SIZE) / frame.rows);
    int width = std::max(1, cvRound(frame.cols * scale));
    int height = std::max(1, cvRound(frame.rows * scale));
    result.scaleX = double(width) / frame.cols;
    result.scaleY = double(height) / frame.rows;
    result.padLeft = (INPUT_SIZE - width) / 2;
    result.padTop = (INPUT_SIZE - height) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(width, height));
    cv::Mat image(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(image(cv::Rect(result.padLeft, result.padTop, width, height)));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int channelSize = INPUT_SIZE * INPUT_SIZE;
    result.pixels.resize(3 * channelSize);
    for (int channel = 0; channel < 3; ++channel) {
        for (int y = 0; y < INPUT_SIZE; ++y) {
            for (int x = 0; x < INPUT_SIZE; ++x) {
                float value = image.at<cv::Vec3b>(y, x)[channel] / 255.0f;
                int index = channel * channelSize + y * INPUT_SIZE + x;
                result.pixels[index] = Ort::Float16_t(value);
            }
        }
    }
    return result;
}

// 3. 推理：返回输出张量，其内存由返回对象管理。
std::vector<Ort::Value> runModel(Ort::Session& session, ModelInput& input,
                                 const char* inputName, const char* outputName) {
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> shape = {1, 3, INPUT_SIZE, INPUT_SIZE};
    auto tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
        memory, input.pixels.data(), input.pixels.size(), shape.data(), shape.size());
    const char* inputNames[] = {inputName};
    const char* outputNames[] = {outputName};
    Ort::RunOptions options;
    auto outputs = session.Run(options, inputNames, &tensor, 1, outputNames, 1);

    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error("Expected one output tensor");
    }
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> expected = {1, PREDICTION_COUNT, PREDICTION_WIDTH};
    if (info.GetShape() != expected ||
        info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Expected float32 output [1,25200,22]");
    }
    return outputs;
}

float sigmoid(float value) {
    if (value >= 0) {
        return 1.0f / (1.0f + std::exp(-value));
    }
    float expValue = std::exp(value);
    return expValue / (1.0f + expValue);
}

int largestIndex(const float* scores, int count) {
    int best = 0;
    for (int i = 1; i < count; ++i) {
        if (scores[i] > scores[best]) {
            best = i;
        }
    }
    return best;
}

// 4. 解析：置信度筛选、类别识别、坐标恢复和 NMS。
std::vector<Detection> decode(const float* predictions, const ModelInput& input) {
    std::vector<Detection> candidates;
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;

    for (int i = 0; i < PREDICTION_COUNT; ++i) {
        const float* row = predictions + i * PREDICTION_WIDTH;
        bool valid = true;
        for (int j = 0; j < PREDICTION_WIDTH; ++j) {
            if (!std::isfinite(row[j])) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }

        float confidence = sigmoid(row[8]);
        if (confidence <= CONFIDENCE_THRESHOLD) {
            continue;
        }
        Detection detection;
        detection.confidence = confidence;
        detection.color = largestIndex(row + 9, 4);
        detection.pattern = largestIndex(row + 13, 9);
        for (int j = 0; j < 4; ++j) {
            double x = (row[2 * j] - input.padLeft) / input.scaleX;
            double y = (row[2 * j + 1] - input.padTop) / input.scaleY;
            detection.corners.emplace_back(x, y);
        }

        double minX = detection.corners[0].x;
        double maxX = minX;
        double minY = detection.corners[0].y;
        double maxY = minY;
        for (const cv::Point2d& point : detection.corners) {
            minX = std::min(minX, point.x);
            maxX = std::max(maxX, point.x);
            minY = std::min(minY, point.y);
            maxY = std::max(maxY, point.y);
        }
        if (maxX <= minX || maxY <= minY) {
            continue;
        }
        detection.box = cv::Rect2d(minX, minY, maxX - minX, maxY - minY);
        candidates.push_back(detection);
        boxes.push_back(detection.box);
        scores.push_back(confidence);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, keep);
    std::vector<Detection> result;
    for (int index : keep) {
        result.push_back(candidates[index]);
    }
    return result;
}

// 5. PnP：选择相机前方、重投影误差最小的平面候选解。
json estimatePose(const Detection& detection, const Calibration& calibration) {
    json result = {
        {"color", COLOR_NAMES[detection.color]},
        {"pattern", PATTERN_NAMES[detection.pattern]},
        {"confidence", detection.confidence},
        {"corners_px", json::array()},
        {"corners_camera_m", nullptr},
        {"reprojection_error_px", nullptr}
    };
    for (const cv::Point2d& point : detection.corners) {
        result["corners_px"].push_back({point.x, point.y});
    }

    double width = 0.135;
    if (detection.pattern == 1 || detection.pattern == 8) {
        width = 0.230;
    }
    double height = 0.056;
    std::vector<cv::Point3d> objectPoints = {
        {-width / 2, -height / 2, 0},
        {-width / 2,  height / 2, 0},
        { width / 2,  height / 2, 0},
        { width / 2, -height / 2, 0}
    };

    try {
        std::vector<cv::Mat> rotations, translations;
        cv::solvePnPGeneric(objectPoints, detection.corners, calibration.camera,
                            calibration.distortion, rotations, translations,
                            false, cv::SOLVEPNP_IPPE);
        double bestError = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < rotations.size(); ++i) {
            if (!cv::checkRange(rotations[i]) || !cv::checkRange(translations[i])) {
                continue;
            }
            cv::Mat rotation;
            cv::Rodrigues(rotations[i], rotation);
            json cameraCorners = json::array();
            bool valid = true;
            for (const cv::Point3d& point : objectPoints) {
                cv::Mat object = (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
                cv::Mat cameraPoint = rotation * object + translations[i];
                if (!cv::checkRange(cameraPoint) || cameraPoint.at<double>(2) <= 0) {
                    valid = false;
                    break;
                }
                cameraCorners.push_back({cameraPoint.at<double>(0),
                                         cameraPoint.at<double>(1),
                                         cameraPoint.at<double>(2)});
            }
            if (!valid) {
                continue;
            }

            std::vector<cv::Point2d> projected;
            cv::projectPoints(objectPoints, rotations[i], translations[i],
                              calibration.camera, calibration.distortion, projected);
            double sum = 0;
            for (int j = 0; j < 4; ++j) {
                cv::Point2d difference = projected[j] - detection.corners[j];
                sum += difference.dot(difference);
            }
            double error = std::sqrt(sum / 4.0);
            if (std::isfinite(error) && error < bestError) {
                bestError = error;
                result["corners_camera_m"] = cameraCorners;
                result["reprojection_error_px"] = error;
            }
        }
    } catch (const cv::Exception&) {
        // 几何退化时保留二维检测，位姿字段保持 null。
    }
    return result;
}

void createParentDirectory(const std::filesystem::path& path) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
}

// 保存第一帧去重后的轮廓，帮助检查坐标是否正确。
void savePreview(const cv::Mat& frame, const std::vector<Detection>& detections,
                 const std::filesystem::path& path) {
    cv::Mat drawing = frame.clone();
    for (const Detection& detection : detections) {
        for (int i = 0; i < 4; ++i) {
            const cv::Point2d& a = detection.corners[i];
            const cv::Point2d& b = detection.corners[(i + 1) % 4];
            // 绘图前限制数值范围；不改动用于 PnP 的原始坐标。
            cv::Point start(cvRound(std::clamp(a.x, -100000.0, 100000.0)),
                            cvRound(std::clamp(a.y, -100000.0, 100000.0)));
            cv::Point end(cvRound(std::clamp(b.x, -100000.0, 100000.0)),
                          cvRound(std::clamp(b.y, -100000.0, 100000.0)));
            cv::line(drawing, start, end, cv::Scalar(0, 255, 0), 2);
            cv::putText(drawing, std::to_string(i), start, cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 0, 255), 1);
        }
    }
    createParentDirectory(path);
    if (!cv::imwrite(path.string(), drawing)) {
        throw std::runtime_error("Cannot save preview");
    }
}

// 6. 主流程：初始化一次模型，依次处理所有视频帧。
int main(int argc, char* argv[]) {
    try {
        Arguments args = readArguments(argc, argv);
        Calibration calibration = loadCalibration(args.calibration);

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vision");
        Ort::SessionOptions options;
        Ort::Session session(env, args.model.c_str(), options);
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
            throw std::runtime_error("Expected one model input and output");
        }
        auto inputType = session.GetInputTypeInfo(0);
        auto inputInfo = inputType.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> expected = {1, 3, INPUT_SIZE, INPUT_SIZE};
        if (inputInfo.GetShape() != expected ||
            inputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            throw std::runtime_error("Expected float16 input [1,3,640,640]");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto outputName = session.GetOutputNameAllocated(0, allocator);

        cv::VideoCapture video(args.video);
        if (!video.isOpened()) {
            throw std::runtime_error("Cannot open video: " + args.video);
        }
        // 防止误把输入文件当作输出路径。
        for (const auto& inputPath : {args.model, std::filesystem::path(args.video),
                                      std::filesystem::path(args.calibration)}) {
            for (const auto& outputPath : {args.output, args.preview}) {
                if (!outputPath.empty() && std::filesystem::weakly_canonical(inputPath) ==
                    std::filesystem::weakly_canonical(outputPath)) {
                    throw std::runtime_error("Output path must differ from input paths");
                }
            }
        }
        if (!args.preview.empty() && std::filesystem::weakly_canonical(args.preview) ==
            std::filesystem::weakly_canonical(args.output)) {
            throw std::runtime_error("Preview and JSONL must use different paths");
        }
        createParentDirectory(args.output);
        std::ofstream output(args.output);
        output.exceptions(std::ios::failbit | std::ios::badbit);

        cv::Mat frame;
        int frameIndex = 0;
        int totalDetections = 0;
        while (video.read(frame)) {
            if (frame.size() != calibration.imageSize) {
                throw std::runtime_error("Video resolution differs from calibration");
            }
            ModelInput input = preprocess(frame);
            auto predictions = runModel(session, input, inputName.get(), outputName.get());
            auto detections = decode(predictions[0].GetTensorData<float>(), input);

            json record = {
                {"frame_index", frameIndex},
                {"timestamp_s", frameIndex / VIDEO_FPS},
                {"detections", json::array()}
            };
            for (const Detection& detection : detections) {
                record["detections"].push_back(estimatePose(detection, calibration));
            }
            output << record.dump() << '\n';
            if (frameIndex == 0 && !args.preview.empty()) {
                savePreview(frame, detections, args.preview);
            }
            totalDetections += static_cast<int>(detections.size());
            ++frameIndex;
            if (frameIndex % 30 == 0) {
                std::cout << "Processed " << frameIndex << " frames\n";
            }
        }
        if (frameIndex == 0) {
            throw std::runtime_error("No readable video frames");
        }
        // 本作业两个片段各有 360 帧，避免将截断读取当作成功。
        if (frameIndex != 360) {
            throw std::runtime_error("Expected 360 assignment frames; check input or decoder");
        }
        output.close();
        std::cout << "Done: " << frameIndex << " frames, "
                  << totalDetections << " detections\n";
        std::cout << "JSONL: " << args.output.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
