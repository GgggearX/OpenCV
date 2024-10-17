import os
import cv2
import numpy as np
import time
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt

# 初始化PaddleOCR，支持中英文混合识别
ocr = PaddleOCR(use_angle_cls=True, lang='ch', drop_score=0.5, show_log=False)

# 确保 output 文件夹存在
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 图像预处理函数（分别展示和保存增强对比度、Sobel边缘检测、二值化的图像）
def preprocess_image(image):
    # 转为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 增强对比度
    contrast_enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    plt.imshow(contrast_enhanced, cmap='gray')
    plt.title("对比度增强")
    plt.show()
    cv2.imwrite(os.path.join(output_dir, 'contrast_enhanced.png'), contrast_enhanced)

    # Sobel边缘检测
    sobelx = cv2.Sobel(contrast_enhanced, cv2.CV_64F, 1, 0, ksize=3)  # X方向
    sobely = cv2.Sobel(contrast_enhanced, cv2.CV_64F, 0, 1, ksize=3)  # Y方向
    sobel_combined = cv2.magnitude(sobelx, sobely)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title("Sobel边缘检测")
    plt.show()
    cv2.imwrite(os.path.join(output_dir, 'sobel_edges.png'), sobel_combined)

    # 二值化
    _, binary = cv2.threshold(contrast_enhanced, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    plt.imshow(binary, cmap='gray')
    plt.title("二值化")
    plt.show()
    cv2.imwrite(os.path.join(output_dir, 'binary_image.png'), binary)

    return binary

# OCR检测并识别文本，保存检测框选出的文字到txt文件
def ocr_detection_and_recognition(image):
    print("开始OCR检测和识别...")

    # 使用 PaddleOCR 进行文本检测和识别
    result = ocr.ocr(image, cls=True)

    recognized_texts = []
    boxes = []
    texts = []
    scores = []
    confidence_threshold = 0.5  # 设定置信度阈值

    # 输出PaddleOCR的结果以供调试
    print(f"OCR 识别结果 (原始数据): {result}")

    # 提取有效的文本框和对应的文本内容
    for line in result:
        for subline in line:  # 遍历每个文本框区域
            box = subline[0]  # 获取文本框的坐标
            text = subline[1][0]  # 获取文本内容
            score = subline[1][1]  # 获取置信度

            if score >= confidence_threshold:  # 置信度过滤
                recognized_texts.append(text)
                boxes.append(box)
                texts.append(text)
                scores.append(score)
                print(f"识别出的文本: {text}, 置信度: {score}")

    # 检查是否有有效文本
    if len(recognized_texts) == 0:
        print("未检测到有效文本。")
    else:
        print(f"检测到的文本数量: {len(recognized_texts)}")

    # 将识别的文本保存为 txt 文件
    with open(os.path.join(output_dir, "recognized_texts.txt"), "w", encoding="utf-8") as f:
        for text in recognized_texts:
            f.write(text + '\n')

    # 返回检测框和文本
    return boxes, texts, scores

# 绘制OCR检测框并输出框选图像
def draw_recognized_text(image, boxes, texts, scores):
    # 使用PaddleOCR的draw_ocr函数在图像上绘制文本框
    boxed_image = draw_ocr(image, boxes, texts, scores)

    # 转换为RGB格式并显示图像
    boxed_image_rgb = cv2.cvtColor(np.array(boxed_image), cv2.COLOR_BGR2RGB)
    plt.imshow(boxed_image_rgb)
    plt.title("检测和识别结果")
    plt.show()

    # 保存带有检测框的图像
    cv2.imwrite(os.path.join(output_dir, 'output_with_boxes.png'), np.array(boxed_image))

# 主流程函数
def main(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    if image is None:
        print(f"图像未能读取，请检查路径或文件是否存在: {image_path}")
        return

    # 记录处理时间
    start_time = time.time()

    # 预处理图像
    preprocessed_image = preprocess_image(image)

    # OCR检测并识别文本
    boxes, texts, scores = ocr_detection_and_recognition(preprocessed_image)

    # 绘制OCR识别结果并保存图像
    draw_recognized_text(image, boxes, texts, scores)

    # 输出总处理时间
    end_time = time.time()
    print(f"总处理时间: {end_time - start_time:.2f} 秒")

# 调用主流程
main('./img.png')
