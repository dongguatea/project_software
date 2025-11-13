import argparse
import csv
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple

from PIL import Image


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir",default="./labeltemplate/ddg-104/1024",help="",)
    parser.add_argument("--img-ext",default=".jpg")
    parser.add_argument("--output-dir",default='./label')

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for TXT files in sub‑directories.",
    )
    return parser.parse_args()


def read_corners(txt_path: pathlib.Path) -> Dict[int, List[Tuple[float, float]]]:
    """Read an annotation file and return a mapping: object_id -> list[(x, y)]."""
    corners_by_id: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    with txt_path.open(encoding="utf-8") as fp:
        # The template uses *tab* separators, but we fall back to any whitespace
        reader = csv.reader(fp, delimiter="\t")
        for row in reader:
            # Expect at least: obj_id, corner_label, x, y (columns 0,2,3)
            if len(row) < 4:
                continue  # skip malformed / empty lines
            try:
                obj_id = int(row[0].strip())
                x = float(row[-2].strip())
                y = float(row[-1].strip())
            except ValueError:
                # Header lines or invalid numbers are ignored
                continue
            corners_by_id[obj_id].append((x, y))

    return corners_by_id


def corners_to_bbox(corners: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """From four corner points compute (cx, cy, w, h) in absolute pixels."""
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    return cx, cy, width, height


def convert_file(txt_path: pathlib.Path, img_ext: str, out_dir: pathlib.Path | None) -> None:
    """Convert a single TXT file and write the YOLO annotation TXT."""

    img_path = txt_path.with_suffix(img_ext)

    # Read image dimensions once
    try:
        w_img, h_img = 1024,1024
    except Exception as exc:
        print(f"[ERROR] Cannot open {img_path}: {exc}")
        return

    corners_by_id = read_corners(txt_path)
    if not corners_by_id:
        print(f"[WARNING] No valid corners in {txt_path.name} → skip")
        return

    yolo_lines: List[str] = []
    for cls_id, corners in corners_by_id.items():
        # Some files may list fewer than four corners; ensure at least two points
        if len(corners) < 2:
            continue
        cx, cy, bw, bh = corners_to_bbox(corners)
        yolo_lines.append(
            f"{cls_id} {cx / w_img:.6f} {cy / h_img:.6f} "
            f"{bw / w_img:.6f} {bh / h_img:.6f}")

    if not yolo_lines:
        print(f"[WARNING] {txt_path.name} produced 0 boxes → skip")
        return

    dest_dir = out_dir or txt_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / txt_path.name
    dest_path.write_text("\n".join(yolo_lines), encoding="utf-8")
    print(f"[OK] {txt_path.name} → {dest_path.relative_to(dest_dir)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    label_root = pathlib.Path(args.label_dir)
    if not label_root.exists():
        raise SystemExit(f"label-dir {label_root} does not exist")

    pattern = "**/*.txt" if args.recursive else "*.txt"
    txt_files = list(label_root.glob(pattern))
    if not txt_files:
        raise SystemExit("No TXT annotation files found.")

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    for txt in txt_files:
        convert_file(txt, args.img_ext, out_dir)

    print("\nAll files processed. Done.")


if __name__ == "__main__":
    main()

# import os
# from pathlib import Path
#
#
# def convert_to_yolo_format(input_file, image_width, image_height):
#     """
#     将给定的标注文件转换为YOLO格式
#     :param input_file: 输入的标注文件路径
#     :param image_width: 图像的宽度
#     :param image_height: 图像的高度
#     :return: YOLO格式的标注数据列表
#     """
#     yolo_annotations = []
#
#     with open(input_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()[1:]  # 跳过表头
#
#         for line in lines:
#             data = line.strip().split('\t')
#
#             if len(data) < 5:  # 数据格式不完整
#                 continue
#
#             try:
#                 target_id = int(data[0])  # 目标ID
#                 # 提取角点坐标（跳过描述性文字如“左下”）
#                 x1, y1 = int(data[2]), int(data[3])  # 左上角
#                 x2, y2 = int(data[4]), int(data[3])  # 右上角
#                 x3, y3 = int(data[4]), int(data[5])  # 右下角
#                 x4, y4 = int(data[2]), int(data[5])  # 左下角
#             except ValueError:  # 捕获无效的数字转换错误
#                 print(f"[ERR] 无效数据行: {line.strip()}")
#                 continue
#
#             # 计算框的中心 (center_x, center_y) 和宽度 (width), 高度 (height)
#             center_x = (x1 + x3) / 2 / image_width
#             center_y = (y1 + y3) / 2 / image_height
#             width = abs(x1 - x3) / image_width
#             height = abs(y1 - y3) / image_height
#
#             # 生成YOLO格式标注（class_id center_x center_y width height）
#             yolo_annotations.append(f"{target_id} {center_x} {center_y} {width} {height}")
#
#     return yolo_annotations
#
#
# def save_yolo_annotations(output_dir, annotations, file_name):
#     """
#     保存YOLO格式的标注数据到文件
#     :param output_dir: 输出目录
#     :param annotations: YOLO格式的标注数据
#     :param file_name: 输出文件名
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, file_name)
#
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for annotation in annotations:
#             f.write(f"{annotation}\n")
#
#
# def process_directory(input_dir, output_dir, image_width, image_height):
#     """
#     处理给定目录中的所有标注文件，并转换为YOLO格式
#     :param input_dir: 输入标注文件夹
#     :param output_dir: 输出YOLO标注文件夹
#     :param image_width: 图像的宽度
#     :param image_height: 图像的高度
#     """
#     for txt_file in Path(input_dir).rglob('*.txt'):
#         annotations = convert_to_yolo_format(txt_file, image_width, image_height)
#         save_yolo_annotations(output_dir, annotations, txt_file.name)
#
#
# # 示例：转换文件夹内所有标注文件为YOLO格式
# input_directory = './labeltemplate/ddg-104/1024'  # 假设标注文件保存在此目录
# output_directory = './label/ddg-104'  # 输出YOLO格式的标注文件
#
# # 假设所有图像的大小为 1920x1080
# image_width = 1024
# image_height = 1024
#
# process_directory(input_directory, output_directory, image_width, image_height)
