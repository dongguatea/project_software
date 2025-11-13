
"""
在 data/ 与 labels/ 目录中查找图像和对应标注文件，
随机打乱后按 8:1:1 划分为 train/val/test 三个子集。

用法:
    python split_yolov12_dataset.py \
        --data-dir ./data \
        --labels-dir ./labels \
        --out-dir ./dataset \
        --image-exts .jpg .png \
        --seed 42
"""

import argparse
import random
from pathlib import Path
import shutil
import sys

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv12 数据集划分脚本")
    p.add_argument('--data-dir',   type=Path, default=Path('./data'),
                   help='存放图像的根目录')
    p.add_argument('--labels-dir', type=Path, default=Path('./labels'),
                   help='存放所有标签 .txt 文件的目录')
    p.add_argument('--out-dir',    type=Path, default=Path('./dataset'),
                   help='输出数据集目录')
    p.add_argument('--image-exts', nargs='+', default=['.jpg', '.jpeg', '.png'],
                   help='需要识别的图像后缀（可多个）')
    p.add_argument('--seed', type=int, default=42, help='随机种子，保证可复现')
    return p.parse_args()

def gather_images(data_dir: Path, image_exts):
    imgs = []
    for ext in image_exts:
        imgs.extend(data_dir.rglob(f'*{ext.lower()}'))
        imgs.extend(data_dir.rglob(f'*{ext.upper()}'))  # 兼容大写
    return imgs

def split_indices(n, train_ratio=0.8, val_ratio=0.1):
    """返回 train/val/test 的索引列表"""
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)
    idx = list(range(n))
    random.shuffle(idx)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]

def copy_pair(img_path: Path, labels_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path):
    """复制图片及对应标签到目标目录（若标签不存在则提示）"""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, dst_img_dir / img_path.name)

    lbl_name = img_path.with_suffix('.txt').name #获取图片除去后缀名后的名称，with_suffix是替换后缀名
    lbl_src  = labels_dir / lbl_name
    if lbl_src.exists():
        shutil.copy2(lbl_src, dst_lbl_dir / lbl_name)
    else:
        print(f'未找到标签文件: {lbl_src}', file=sys.stderr)

def main():
    args = parse_args()
    random.seed(args.seed)

    images = gather_images(args.data_dir, args.image_exts)
    if not images:
        sys.exit(f'在 {args.data_dir} 内未找到任何图像文件，脚本终止。')

    # 划分索引
    tr_idx, val_idx, te_idx = split_indices(len(images))

    subsets = {
        'train': [images[i] for i in tr_idx],
        'val':   [images[i] for i in val_idx],
        'test':  [images[i] for i in te_idx],
    }

    # 复制文件
    for subset_name, subset_imgs in subsets.items():
        img_out_dir = args.out_dir / subset_name / 'images'
        lbl_out_dir = args.out_dir / subset_name / 'labels'
        for img_path in subset_imgs:
            copy_pair(img_path, args.labels_dir, img_out_dir, lbl_out_dir)

    # # 简要统计
    # print('划分完成:')
    # for k, v in subsets.items():
    #     print(f'  {k:<5}: {len(v)} 张图片')

if __name__ == '__main__':
    main()
