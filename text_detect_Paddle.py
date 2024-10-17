import os
import time  # 引入time模块用于计时
import numpy as np
from paddleocr import PaddleOCR  # 导入PaddleOCR库
import cv2

# 初始化OCR模型，启用文本识别
# use_angle_cls=True 表示启用文本方向分类
# lang='ch' 表示使用中文语言包
# rec=True 表示启用文本识别功能
# drop_score=0.3 表示置信度低于0.3的识别结果会被丢弃
# show_log=False 关闭PaddleOCR的日志输出
ocr = PaddleOCR(use_angle_cls=True, lang='ch', rec=True, drop_score=0.3, show_log=False)


def detect_identify_text(image, img_path):
    overall_start_time = time.time()  # 记录开始时间

    # 计时：检查图像加载
    if image is None:
        print(f"无法读取图像")
    else:
        # 计时：OCR文本检测和识别
        step_start_time = time.time()
        print("PaddleOCR检测+识别：")
        # 使用PaddleOCR的ocr方法进行文本检测和识别，cls=True 启用文本方向分类
        result = ocr.ocr(img_path, cls=True)
        step_end_time = time.time()
        print(f"OCR文本检测和识别时间: {step_end_time - step_start_time:.2f} 秒")

        # 检查OCR结果是否为空或无效
        if result is None or not result or len(result) == 0 or result[0] is None:
            print("未检测到任何文本区域")
        else:
            # 计时：提取文本信息
            step_start_time = time.time()
            boxes = [elements[0] for elements in result[0]]  # 提取边界框信息（位置信息）
            texts = [elements[1][0] for elements in result[0]]  # 提取OCR识别的文本
            scores = [elements[1][1] for elements in result[0]]  # 提取OCR识别置信度
            step_end_time = time.time()
            print(f"提取文本信息时间: {(step_end_time - step_start_time) * 1000:.2f} 毫秒")

            step_start_time = time.time()
            # 遍历所有检测到的边界框
            for idx, box in enumerate(boxes):
                # 将边界框坐标转换为整数
                box = np.array(box).astype(int)
                # 直接绘制边界框轮廓，不需要背景
                cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
                # 标出编号（如“text1,text2”等）
                put_text_label(image, box, f"text {idx + 1}")  # 在每个框上标记对应的文本编号
            step_end_time = time.time()
            print(f"绘制边界框时间: {(step_end_time - step_start_time) * 1000:.2f} 毫秒")

            # 保存带有边界框的图像
            output_path = '../result_images/paddle_detected.png'
            cv2.imwrite(output_path, image)

            # 计时：保存识别文本
            output_text_folder = '../text_output/'
            os.makedirs(output_text_folder, exist_ok=True)  # 如果文件夹不存在则创建

            output_text_path = os.path.join(output_text_folder, 'paddle_recognized_text.txt')
            with open(output_text_path, 'w', encoding='utf-8') as f:
                for idx, (text, score) in enumerate(zip(texts, scores)):
                    f.write(f"text {idx + 1}: {text}\n")
            print(f"检测到的文本已保存到 {output_text_path}")

            # 打印检测到的文本和置信度
            for idx, text in enumerate(texts):
                print(f"text {idx + 1}: {text} (置信度: {scores[idx]:.2f})")

    # 记录总的执行时间
    overall_end_time = time.time()
    print(f"文本检测、识别总执行时间: {overall_end_time - overall_start_time:.2f} 秒")
    print("PaddleOCR检测识别完毕")
    # 显示带有边界框的图像

def put_text_label(image, box, text):
    """
    在边界框的左上角绘制对应的文本编号，字体大小根据边界框大小自动调整。
    :param image: 输入的图像
    :param box: 边界框坐标
    :param text: 要显示的文本编号
    """
    # 获取边界框的左上角坐标
    min_x, min_y = np.min(box[:, 0]), np.min(box[:, 1])
    max_x, max_y = np.max(box[:, 0]), np.max(box[:, 1])

    # 计算边界框的高度和宽度
    box_height = max_y - min_y

    # 根据边界框的高度动态调整字体大小比例
    font_scale = box_height / 75

    # 设置文本的字体、颜色和粗细
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
    font_color = (255, 204, 0)
    font_thickness = int(font_scale * 2)  # 字体粗细根据比例调整

    # 在图像上绘制文本编号
    cv2.putText(image, text, (min_x, min_y - 10), font, font_scale, font_color, font_thickness)
