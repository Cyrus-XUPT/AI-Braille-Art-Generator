import sys
import numpy as np
from PIL import Image
import torch
from controlnet_aux import LineartDetector

# [1] 盲文映射的核心矩阵逻辑
BRAILLE_BASE = 0x2800
DOT_WEIGHTS = np.array([[1, 8], [2, 16], [4, 32], [64, 128]], dtype=np.uint16)
BRAILLE_TABLE = np.array([chr(BRAILLE_BASE + i) for i in range(256)])

def resize_for_braille(img: Image.Image, width_chars: int) -> Image.Image:
    """缩放到适合微信的字符宽度 (1字符 = 2像素宽)"""
    target_w = width_chars * 2
    orig_w, orig_h = img.size
    target_h = max(4, round(orig_h * target_w / orig_w))
    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

def render_braille(dots: np.ndarray) -> str:
    """利用 NumPy 将布尔矩阵按 2x4 切片映射为盲文字符"""
    h, w = dots.shape
    pad_h = (-h) % 4
    pad_w = (-w) % 2
    if pad_h or pad_w:
        dots = np.pad(dots, ((0, pad_h), (0, pad_w)), mode="edge")

    blocks = dots.reshape(dots.shape[0] // 4, 4, dots.shape[1] // 2, 2)
    weights = DOT_WEIGHTS[None, :, None, :]
    codes = (blocks * weights).sum(axis=(1, 3), dtype=np.uint16)
    return "\n".join("".join(row) for row in BRAILLE_TABLE[codes])

# [2] 深度学习：特征图提取与转换
def main():
    # 替换为你刚才测试失败的那张复杂的高清真人照片
    image_path = "curry_line.jpg" 
    width_chars = 60 # 微信安全宽度

    print("正在加载预训练的 CNN 线稿提取模型...")
    print("(初次运行会从 HuggingFace 自动下载权重，大约几百MB，请耐心等待)")
    
    # 实例化 ControlNet 级别的语义边缘提取器
    model = LineartDetector.from_pretrained("lllyasviel/Annotators")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"模型加载完毕 ({device})。正在分析图像: {image_path}")
    original_img = Image.open(image_path).convert("RGB")

    # 核心降维打击：通过神经网络推理，将复杂光影转化为纯净的黑底白线
    lineart_img = model(original_img)
    
    # 将模型输出的特征图转为灰度矩阵，并缩小到 48 像素宽
    gray_lineart = lineart_img.convert("L")
    small_gray = resize_for_braille(gray_lineart, width_chars)
    pixels = np.asarray(small_gray, dtype=np.uint8)

    # 提取有效线段：神经网络输出的是黑底白线，因此“亮像素”就是我们需要凸起的盲文点
    # 因为 CNN 提纯过的线条极度纯粹，直接用一个极低的阈值即可完美分离
    dots = pixels > 40

    print("语义特征提取完毕！正在执行张量映射...")
    art = render_braille(dots)

    print("\n最终生成的 AI 盲文字符画：\n")
    print(art)

    with open("ai_output.txt", "w", encoding="utf-8") as f:
        f.write(art + "\n")
    print("\n[成功] 已保存到 ai_output.txt，打开全选即可复制发送！")

if __name__ == "__main__":
    main()
