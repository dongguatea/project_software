#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset.py
将 images 目录下的所有图片路径按 8:1:1 比例随机划分为
train.txt / val.txt / test.txt（绝对路径，逐行存放）。
目录结构示例:
images/
 ├── manbo/
 │    ├── 1/
 │    │    ├── 00001.bmp
 │    │    └── ...
 │    ├── 2/
 │    └── ...
 └── panmao/
      ├── 1/
      └── ...
"""

from pathlib import Path
import random
import os
import math

# 修改这里可以替换根目录
# ROOT_DIR = Path(__file__).resolve().parent #先获取当前文件的绝对路径，然后找到它所在的上一级路径
ROOT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = ROOT_DIR / "images"

# 1. 收集所有 bmp 图像路径
all_images = [p.resolve() for p in IMAGES_DIR.rglob("*.bmp")]
if not all_images:
    raise RuntimeError(f"在 {IMAGES_DIR} 下未找到任何 .bmp 文件，请检查目录结构。")

# 2. 随机打乱路径列表（不影响实际文件）
random.shuffle(all_images)

# # 3. 按 8:1:1 划分
# n_total = len(all_images)
# n_train = math.floor(n_total * 0.8)
# n_val   = math.floor(n_total * 0.1)
# # 剩余部分全部归入 test
# n_test  = n_total - n_train - n_val

# train_paths = all_images[:n_train]
# val_paths   = all_images[n_train:n_train + n_val]
# test_paths  = all_images[n_train + n_val:]

# # 4. 写入 txt 文件（逐行一个绝对路径）
# def write_paths(paths, filename):
#     filepath = os.path.join(ROOT_DIR, filename)
#     Path(filepath).write_text("\n".join(map(str, paths)), encoding="utf-8")

# write_paths(train_paths, "train.txt")
# write_paths(val_paths,   "val.txt")
# write_paths(test_paths,  "test.txt")

# print(f"已生成数据集划分："
#       f"train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

# 3. 按8：2划分
n_total = len(all_images)
n_train = math.floor(n_total * 0.8)
n_val = math.floor(n_total * 0.2)

train_paths = all_images[:n_train]
val_paths = all_images[n_train:]

# 4. 写入txt文件
def write_paths(paths,filename):
    filepath = ROOT_DIR / filename
    filepath.write_text("\n".join(map(str, paths)), encoding="utf-8")
write_paths(train_paths, "train.txt")
write_paths(val_paths, "val.txt")
print(f"已生成数据集划分：train={len(train_paths)}, val={len(val_paths)}")