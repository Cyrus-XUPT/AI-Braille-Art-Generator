# AI-Braille-Art-Generator (深度语义字符画生成器)

基于深度学习与 NumPy 张量映射的高精度“图像 -> 盲文字符画”转换工具。

## 💡 项目背景
传统的字符画生成工具通常依赖 OpenCV 或 PIL 进行简单的灰度阈值切分（如 Otsu 大津法）。但在面对真实摄影照片或带有复杂铅笔光影的线稿时，传统算法由于缺乏**空间语义理解能力**，往往会导致线条断裂、背景噪点过多、微缩分辨率下细节糊成一团等问题。

本项目引入了计算机视觉领域的卷积神经网络（CNN），通过预训练的 `LineartDetector` 模型，先对高维图像进行“语义降维打击”，剥离背景光影，提取极其纯净的线稿特征张量，再通过底层 NumPy 矩阵运算无缝映射为 Unicode 盲文字符，从而实现对复杂图像的“神还原”。

## 🚀 核心特性
- **语义级边缘提取**：利用深度学习模型，精准识别主体轮廓，无视光影与渐变干扰。
- **矩阵极速映射**：摒弃低效的 `for` 循环遍历，利用 NumPy 的 `reshape` 和矩阵点乘，将布尔矩阵以 `2x4` 切片瞬间映射为盲文点阵。
- **破除分辨率封印**：灵活自定义输出宽度，完美适配微信防折行（24字符）或抖音/B站宽屏展示（50+字符）。

## 🛠️ 环境配置
本项目支持 CPU 快速推理。推荐在 Ubuntu 环境下使用 `venv` 或 `conda` 运行。
```bash
# 1. 安装精简版 PyTorch (CPU)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# 2. 安装视觉依赖与矩阵库 (推荐使用国内镜像加速)
pip install controlnet_aux numpy pillow -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
