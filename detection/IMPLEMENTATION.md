# 检测路线：安装与复现

## 实现范围

本项目完成检测路线，不包含瞄准、跨帧跟踪或硬件控制。CPU 离线处理即可。
入口是 `src/main.cpp`，CMake 可执行目标是 `vision_detection`。

## Windows + CLion 环境

1. 安装 Git、CLion，以及 Visual Studio Build Tools 的“使用 C++ 的桌面开发”（MSVC x64 和 Windows SDK）。
2. 从自己的 GitHub 仓库克隆项目，用 CLion 打开 `detection` 文件夹。
3. 下载 vcpkg：`git clone https://github.com/microsoft/vcpkg.git`，进入该目录执行 `bootstrap-vcpkg.bat`。
4. 从 https://github.com/microsoft/onnxruntime/releases 下载 Windows x64 CPU 发布包，解压到一个包含 `include` 和 `lib` 的目录。
5. CLion Toolchains 选择 Visual Studio/MSVC、amd64。不要与 MinGW 库混用。
6. CMake 配置使用 Release、Ninja，构建目录 `cmake-build-release`。填入下面三个 CMake options，将两个示例绝对路径换成自己安装目录：

```text
-DCMAKE_TOOLCHAIN_FILE=C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DONNXRUNTIME_ROOT=C:/tools/onnxruntime
```

CMake 会按 vcpkg.json 安装 OpenCV、yaml-cpp 和 nlohmann-json。ONNX Runtime 使用预编译包，CMake 将运行所需 DLL 复制到可执行文件旁。
每个 CLion Profile 应使用不同构建目录。改过编译器后重置 CMake 缓存。

## 命令行编译

在 **x64 Native Tools Command Prompt for Visual Studio** 中进入 `detection`，确保 cmake 和 ninja 在 PATH 中：

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DONNXRUNTIME_ROOT=C:/tools/onnxruntime
cmake --build build --target vision_detection
```

若使用 CLion 内置 CMake/Ninja，可在 CLion 内构建，无需另外配置终端 PATH。
素材应位于 `assets`，其中 ONNX、MP4 必须是真实文件而不是 Git LFS 指针。

## 运行两段视频

工作目录为 `detection`。CLion 构建目录使用以下 PowerShell 命令：

```powershell
.\cmake-build-release\vision_detection.exe --model assets/0526.onnx --video assets/small.mp4 --calibration assets/calibration.yaml --output results/small.jsonl --preview results/small_preview.png
.\cmake-build-release\vision_detection.exe --model assets/0526.onnx --video assets/large.mp4 --calibration assets/calibration.yaml --output results/large.jsonl --preview results/large_preview.png
```

命令行使用 `build` 编译时，将可执行文件路径改为 `.\build\vision_detection.exe`。
CLion Program arguments 只填写 `--model ...` 之后的参数，不填写可执行文件名。

也兼容旧的三个位置参数：`assets/0526.onnx assets/small.mp4 assets/calibration.yaml`，默认输出到 results/视频名.jsonl 和 results/视频名_preview.png。
`--preview` 可选，只保存第一帧的去重轮廓与角点编号；无检测时保存原图。
输出文件如已存在会覆盖。输入文件与输出文件不能使用相同路径。

## 算法约定

- 居中 letterbox 到 640×640，填充值 114；BGR 转 RGB、除以 255、转换为 CHW FP16。
- 模型输入 `[1,3,640,640]`，输出 `[1,25200,22]`；启动和推理时检查形状、类型。
- confidence 为 sigmoid(objectness)，不乘颜色或图案分数。
- 置信度采用大于 0.5；NMS IoU 阈值 0.5。先使用作业建议基线，未宣称为最优参数。
- 颜色和图案按作业给定顺序 argmax，保留全部颜色；NMS 不区分类别。
- 四角点恢复到原始像素坐标，不重排：左上、左下、右下、右上。
- 图案 1、base_big 使用 0.230×0.056 m，其余使用 0.135×0.056 m。尺寸不由视频名称决定。
- IPPE 求取平面候选解，排除非有限值及任一角点 Z≤0 的解，在其余解中选重投影 RMS 最小者。
- 每个三维角点为 R×物体角点+t；坐标系为相机右、下、前，单位米。
- PnP 失败保留二维检测，三维角点和误差为 null。重投影误差为 sqrt(四个二维误差平方和/4)。

## 输出与限制

每帧写一条 JSON，无检测也写空数组。帧编号从 0 开始，时间为 index/30，片段分别从零计时。
程序针对作业给定的 30 FPS、360 帧片段；读取帧数不是 360 时返回失败，可能留下部分 JSONL，不能当作完整结果提交。
不进行时间连续性消歧，不人为设置未经验证的 PnP 误差阈值；较大的重投影误差会原样报告，供检查。
没有人工真值标签，检测数量不是准确率，低重投影误差也不证明类别或物理尺寸一定正确。

## 提交

仓库已忽略 IDE、虚拟环境、构建目录和 results。无需提交依赖二进制或虚拟环境。
提交代码、CMakeLists.txt、vcpkg.json 和说明文档。平台只需提交自己的公开仓库 URL。
允许 AI 辅助；本实现使用 AI 协助编写和检查，提交者仍应理解并能定位问题。

## 本次验证记录（2026-10-29）

- MSVC x64 Release 编译通过；两段视频均完整运行，退出码为 0。
- small：360 帧、396 条检测；large：360 帧、151 条检测。
- 全量检查 JSONL 可解析、编号连续、时间戳正确、类别合法、四个二维角点有限、有效三维角点均在相机前方。
- small 抽查帧 0、38、90、180、270、359；large 抽查帧 0、90、137、180、270、359。
- 抽查中可见正常贴近灯条的轮廓，也存在无检测帧；未发现整体坐标缩放或补边偏移。
- large 第 137 帧的轮廓明显偏斜，重投影 RMS 约 11.03 px，不能视为高质量位姿。当前保留该数值，没有按特定帧硬编码修正。
- 类别预测仍可能错误：代码按模型 argmax 选择几何尺寸，不根据文件名或人工观察更改类别。
- 抽查不等于完整人工标注评测；没有准确率或召回率结论。后续可增加可配置位姿质量门限、进一步检查模型类别稳定性，但不能仅凭小重投影误差判断识别正确。
