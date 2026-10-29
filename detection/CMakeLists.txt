cmake_minimum_required(VERSION 3.24)

project(vision_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS
        core
        imgproc
        imgcodecs
        videoio
        highgui
        calib3d
        dnn
)

find_package(yaml-cpp CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

set(ONNXRUNTIME_ROOT "" CACHE PATH
        "Directory containing ONNX Runtime include and lib")

find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS "${ONNXRUNTIME_ROOT}/include"
        NO_DEFAULT_PATH
        REQUIRED
)

find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib"
        NO_DEFAULT_PATH
        REQUIRED
)

add_executable(vision_detection
        src/main.cpp
)

target_include_directories(vision_detection PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(vision_detection PRIVATE
        ${OpenCV_LIBS}
        yaml-cpp::yaml-cpp
        nlohmann_json::nlohmann_json
        ${ONNXRUNTIME_LIBRARY}
)

if(MSVC)
    target_compile_options(vision_detection PRIVATE /utf-8)
endif()

if(WIN32)
    file(GLOB ONNXRUNTIME_DLLS
            "${ONNXRUNTIME_ROOT}/lib/*.dll"
    )

    foreach(runtime_dll IN LISTS ONNXRUNTIME_DLLS)
        add_custom_command(TARGET vision_detection POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${runtime_dll}"
                "$<TARGET_FILE_DIR:vision_detection>"
        )
    endforeach()
endif()
