#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将每3张图片共用的模板标注文件转换为 YOLO 格式，并为每张图片生成独立的标注文件。

假设：
- 输入文件夹中每 3 张图片对应同名的一个模板标注文件（.txt）。
- 注释文件与图片按文件名排序后按顺序一一对应（每三个图片文件对应一个 .txt 文件）。

运行示例：
python batch_convert_yolo.py --dir path/to/folder --class-id 0 --img-ext .jpg .png
"""

import os
import re
import argparse
from PIL import Image

labels_dir = './labels'
os.makedirs(labels_dir, exist_ok=True)
def parse_template(path):
    """解析模板文件，返回每 4 个点一组的边界框列表 [(x_min, y_min, x_max, y_max), ...]"""
    pts = []
    pattern = re.compile(r'宽方向[:：]\s*(\d+\.?\d*)\s*高方向[:：]\s*(\d+\.?\d*)')
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                x, y = float(m.group(1)), float(m.group(2))
                pts.append((x, y))
    bboxes = []
    for i in range(0, len(pts), 4):
        quad = pts[i:i+4]
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        bboxes.append((min(xs), min(ys), max(xs), max(ys)))
    return bboxes


def to_yolo(box, img_w, img_h):
    """将 (x_min, y_min, x_max, y_max) 转为 YOLO 归一化格式 (x_center, y_center, w, h)"""
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return x_center, y_center, w, h


def main():
    parser = argparse.ArgumentParser(description="批量转换 YOLO 标注文件")
    parser.add_argument('--dir', default=r'./images', help="包含图片和标注文件的文件夹路径")
    parser.add_argument('--class-id', type=int, default=0, help="YOLO 类别 ID，默认 0")
    parser.add_argument('--img-ext', nargs='+', default=['.jpg', '.png', '.jpeg'],
                        help="允许的图片扩展名列表，默认为 .jpg .png .jpeg")
    args = parser.parse_args()

    files = sorted(os.listdir(args.dir))
    img_files = [f for f in files if os.path.splitext(f)[1].lower() in args.img_ext]
    label_files = [f for f in files if os.path.splitext(f)[1].lower() == '.txt']

    if len(img_files) % 3 != 0 or len(label_files) != len(img_files) // 3:
        print("警告：图片数量与标注文件数量不匹配，请检查文件命名及数量。")

    for idx, label in enumerate(label_files):
        label_path = os.path.join(args.dir, label)
        bboxes = parse_template(label_path)
        group_imgs = img_files[idx*3:(idx+1)*3]
        img_w, img_h = 512,512
        for img_name in group_imgs:
            img_path = os.path.join(args.dir, img_name)
            yolo_lines = []
            for box in bboxes:
                x_c, y_c, w, h = to_yolo(box, img_w, img_h)
                yolo_lines.append(f"{args.class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

            name = os.path.splitext(os.path.basename(img_path))[0]
            out_txt = os.path.join(labels_dir, name + '.txt')

            with open(out_txt, 'w', encoding='utf-8') as fo:
                fo.write('\n'.join(yolo_lines))
            print(f"生成：{out_txt}")


if __name__ == '__main__':
    main()
