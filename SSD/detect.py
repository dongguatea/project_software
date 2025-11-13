#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSD 批量检测并写入 MySQL（宽松匹配 image_id）
- 递归扫描 --images_dir（多级子目录、.bmp/.png/.jpg/.jpeg）
- 镜像到 --labels_dir 读取 YOLO 标签（class cx cy w h，归一化）
- 用 torchvision.models.detection.ssd300_vgg16 推理
- IoU 贪心匹配计算 TP/FP/FN/F1（无标签时给“代理F1”）
- 将结果批量写入 eval_metrics（或 eval_mertics），并自动创建列 model (0=YOLO, 1=SSD)
- 与数据库的 image_id 关联：以 “小写绝对路径” 为主键；同时兼容相对路径/带 images 前缀/仅文件名的兜底
"""

import os
import re
import gc
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="mysql.connector")

import numpy as np
from PIL import Image

import torch
import torchvision.ops as tvops
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import ssd300_vgg16
from torchvision.models import VGG16_Weights

import mysql.connector


# =========================
# 默认配置（可用命令行覆盖）
# =========================
DB_Config = {
    "host": "localhost",
    "user": "root",
    "password": "asd515359",
    "database": "config_db",
    "port": 3306,
}

MODEL_ID = 1  # 0=YOLO, 1=SSD
BATCH_COMMIT = 5000
CONF_THR = 0.30
NMS_IOU = 0.50
MATCH_IOU = 0.50
PROXY_F1_MODE = "mean_conf"  # 'mean_conf' or 'top3_conf'


# =========================
# 工具函数：路径与标签
# =========================
def normkey(p: str | Path) -> str:
    """统一成小写的绝对路径（正斜杠）"""
    return str(Path(p).resolve()).replace("\\", "/").lower()


def to_label_path(image_abs: str, images_dir: str, labels_dir: str) -> str:
    """
    将图片“绝对路径”转换为标签路径：
    - 先对 images_dir 求相对路径
    - 镜像到 labels_dir 并改后缀为 .txt
    """
    p = Path(image_abs).resolve()
    rel = p.relative_to(Path(images_dir).resolve())
    return str((Path(labels_dir) / rel).with_suffix(".txt")).replace("\\", "/")


def xywhn_to_xyxy(xywhn, img_w, img_h):
    x, y, w, h = xywhn
    return [(x - w/2) * img_w, (y - h/2) * img_h, (x + w/2) * img_w, (y + h/2) * img_h]


def read_gt_boxes(label_path: str, img_w: int, img_h: int, keep_classes: Optional[List[int]] = None):
    gts = []
    if not os.path.isfile(label_path):
        return gts
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue
            cls_id = int(float(p[0]))
            if (keep_classes is not None) and (cls_id not in keep_classes):
                continue
            x, y, w, h = map(float, p[1:])
            gts.append({"cls": cls_id, "box": xywhn_to_xyxy([x, y, w, h], img_w, img_h)})
    return gts


# =========================
# IoU / 贪心匹配
# =========================
def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    b_area = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds, gts, iou_thr):
    """按同类、IoU>=阈值做贪心一对一匹配"""
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if p["cls"] != g["cls"]:
                continue
            iou = box_iou(p["box"], g["box"])
            if iou >= iou_thr:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    matched_p, matched_g, matches = set(), set(), []
    for iou, pi, gi in pairs:
        if pi not in matched_p and gi not in matched_g:
            matched_p.add(pi)
            matched_g.add(gi)
            matches.append((pi, gi, iou))
    un_p = [i for i in range(len(preds)) if i not in matched_p]
    un_g = [i for i in range(len(gts)) if i not in matched_g]
    return matches, un_p, un_g


# =========================
# 模型与推理
# =========================
def build_ssd_model(num_classes: int, backbone_choice: str = "features"):
    """
    backbone_choice: 'features' -> VGG16_Weights.IMAGENET1K_FEATURES
                     'v1'       -> VGG16_Weights.IMAGENET1K_V1
                     'none'     -> 不加载预训练骨干
    """
    if backbone_choice == "features":
        wb = VGG16_Weights.IMAGENET1K_FEATURES
    elif backbone_choice == "v1":
        wb = VGG16_Weights.IMAGENET1K_V1
    else:
        wb = None

    return ssd300_vgg16(
        weights=None,                # 不加载 COCO 检测头
        weights_backbone=wb,         # 加载/不加载 VGG16 骨干
        num_classes=num_classes
    )


@torch.no_grad()
def ssd_infer_one(model, device, img_pil: Image.Image,
                  conf_thr: float, nms_iou: float,
                  classes_keep: Optional[List[int]] = None,
                  label_shift: int = 1):
    """
    返回 preds: [{'cls': yolo_cls_id, 'box': [x1,y1,x2,y2]}, ...], confs: [score...]
    SSD 的 labels: 背景=0，前景从1开始；YOLO 从0开始，所以 yolo_id = ssd_label - label_shift
    """
    img_tensor = to_tensor(img_pil).to(device)  # [0,1] CHW
    out = model([img_tensor])[0]
    boxes = out["boxes"].detach()
    scores = out["scores"].detach()
    labels = out["labels"].detach()

    # 阈值与 NMS
    keep = scores >= conf_thr
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.numel() > 0:
        keep_idx = tvops.nms(boxes, scores, nms_iou)
        boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]

    preds, confs = [], []
    W, H = img_pil.size
    for b, s, l in zip(boxes, scores, labels):
        cls_ssd = int(l.item())
        if cls_ssd == 0:
            continue
        yolo_cls = cls_ssd - label_shift
        if (classes_keep is not None) and (yolo_cls not in classes_keep):
            continue
        x1, y1, x2, y2 = [float(v) for v in b.tolist()]
        x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
        y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
        preds.append({"cls": yolo_cls, "box": [x1, y1, x2, y2]})
        confs.append(float(s.item()))
    return preds, confs


# =========================
# DB：确保列存在 & 批量写入
# =========================
def ensure_model_column(conn, db_name: str, table_candidates: List[str]) -> str:
    """
    返回实际使用的表名；会在不存在时添加列 model TINYINT(1) DEFAULT 0
    优先选择存在的第一个表名（eval_metrics 或 eval_mertics）
    """
    cur = conn.cursor()
    table_name: Optional[str] = None
    for t in table_candidates:
        cur.execute("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s
        """, (db_name, t))
        if cur.fetchone()[0] > 0:
            table_name = t
            break
    if table_name is None:
        raise RuntimeError("既未找到表 'eval_metrics'，也未找到表 'eval_mertics'。")

    # 列检查
    cur.execute("""
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND COLUMN_NAME='model'
    """, (db_name, table_name))
    has_col = cur.fetchone()[0] > 0
    if not has_col:
        cur.execute(f"ALTER TABLE `{table_name}` ADD COLUMN `model` TINYINT(1) NOT NULL DEFAULT 0")
        conn.commit()
        print(f"[DB] 已为 {table_name} 增加列 model (TINYINT, DEFAULT 0)")
    else:
        print(f"[DB] 列 model 已存在于 {table_name}")
    cur.close()
    return table_name


def insert_batch(cur, table_name: str, batch_rows: List[Tuple]):
    sql = f"""
    INSERT INTO `{table_name}`
    (image_id, conf_thr, nms_iou, match_iou, tp, fp, fn, n_gt, f1, model)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    cur.executemany(sql, batch_rows)


# =========================
# 宽松匹配 image_id
# =========================
def build_image_id_resolver(db_rows, images_dir: str):
    """
    构建从“多种路径表达”到 image_id 的解析器
    - 主映射：小写绝对/相对路径（正斜杠）
    - 兜底：仅文件名 -> [多个 id]（若同名多个则放弃）
    """
    key2id: Dict[str, int] = {}
    name2ids: Dict[str, List[int]] = defaultdict(list)

    # 将 DB 的 image_path 规范化后建立索引
    for image_id, image_path in db_rows:
        k = str(image_path).replace("\\", "/")
        key2id[k.lower()] = int(image_id)
        name2ids[Path(k).name.lower()].append(int(image_id))

    images_dir_norm = normkey(images_dir)

    def resolve(image_abs_norm: str) -> Optional[int]:
        # 1) 绝对路径完全匹配
        if image_abs_norm in key2id:
            return key2id[image_abs_norm]

        # 2) 相对 images_dir 的相对路径匹配（含/不含 images/ 前缀）
        try:
            rel = str(Path(image_abs_norm).resolve().relative_to(Path(images_dir_norm).resolve())).replace("\\", "/").lower()
            if rel in key2id:
                return key2id[rel]
            alt = f"images/{rel}"
            if alt in key2id:
                return key2id[alt]
        except Exception:
            pass

        # 3) 仅文件名兜底（同名多条则返回 None）
        fname = Path(image_abs_norm).name.lower()
        ids = name2ids.get(fname, [])
        if len(ids) == 1:
            return ids[0]
        return None

    return resolve


# =========================
# 主程序
# =========================
def main():
    parser = argparse.ArgumentParser(description="SSD batch detect & write to MySQL (robust image_id matching)")
    parser.add_argument("--weights", type=str, required=True, help="SSD 训练好的权重路径（.pth）")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")  # e.g., "cuda:0" or "cpu"
    parser.add_argument("--conf_thr", type=float, default=CONF_THR)
    parser.add_argument("--nms_iou", type=float, default=NMS_IOU)
    parser.add_argument("--match_iou", type=float, default=MATCH_IOU)
    parser.add_argument("--classes_keep", type=str, default="")  # "0,1"
    parser.add_argument("--batch_commit", type=int, default=BATCH_COMMIT)
    parser.add_argument("--backbone", type=str, default="features", choices=["features", "v1", "none"])
    parser.add_argument("--num_classes", type=int, default=0, help="如不为0，则强制作为 (背景+前景类) 的总数；否则从权重里推断")
    args = parser.parse_args()

    # 设备
    device = torch.device(args.device if args.device == "cpu" else (args.device))
    print(f"[INFO] device={device}")

    # 读取权重（先只读 meta，确定类别数）
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"[ERR] weights not found: {weights_path}  (建议用正斜杠 E:/.../xxx.pth)")
    ckpt_cpu = torch.load(str(weights_path), map_location="cpu")
    if isinstance(ckpt_cpu, dict) and "classes" in ckpt_cpu and args.num_classes <= 0:
        num_classes = 1 + len(ckpt_cpu["classes"])
    elif args.num_classes > 0:
        num_classes = args.num_classes
    else:
        # 无法判断时，默认 3（背景+2类）
        num_classes = 3
    print(f"[INFO] num_classes = {num_classes}")

    # 构建模型并加载权重
    model = build_ssd_model(num_classes=num_classes, backbone_choice=args.backbone).to(device)
    state = ckpt_cpu["model"] if (isinstance(ckpt_cpu, dict) and "model" in ckpt_cpu) else ckpt_cpu
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing: print("  missing keys (showing up to 10):", missing[:10])
        if unexpected: print("  unexpected keys (up to 10):", unexpected[:10])
    model.eval()

    # 收集图片
    images_dir = args.images_dir
    labels_dir = args.labels_dir
    img_files = []
    img_files.extend(Path(images_dir).rglob("*.bmp"))
    img_files.sorted()
    if not img_files:
        raise FileNotFoundError(f"未在 {images_dir} 下找到图片")

    fs_abs_keys = [normkey(p) for p in img_files]

    # 连接 DB 并准备表
    conn = mysql.connector.connect(**DB_Config)
    cur = conn.cursor()
    eval_table = ensure_model_column(conn, DB_Config["database"], ["eval_metrics", "eval_mertics"])

    # 拉取 images 表并构建解析器
    cur.execute("SELECT image_id, image_path FROM images")
    db_rows = cur.fetchall()
    resolve_image_id = build_image_id_resolver(db_rows, images_dir)

    # 预检缺失
    image_id_map: Dict[str, int] = {}
    missing = []
    for abs_norm in fs_abs_keys:
        iid = resolve_image_id(abs_norm)
        if iid is None:
            missing.append(abs_norm)
        else:
            image_id_map[abs_norm] = iid

    if missing:
        miss_file = Path("missing_image_paths.txt")
        miss_file.write_text("\n".join(missing), encoding="utf-8")
        cur.close()
        conn.close()
        raise RuntimeError(
            f"images 表缺少 {len(missing)} 条路径（已写 {miss_file}）。\n"
            f"可将这些“绝对路径”或它们的“相对 {images_dir} 的路径”（或加上 images/ 前缀）写入 images.image_path。"
        )

    # 类别过滤
    classes_keep = None
    if args.classes_keep.strip():
        classes_keep = [int(x) for x in args.classes_keep.split(",") if x.strip().isdigit()]
        print(f"[INFO] 仅保留类别: {classes_keep}")

    # 推理与写库
    batch_rows: List[Tuple] = []
    t0 = time.time()
    n_done = 0

    for abs_norm in fs_abs_keys:
        image_id = image_id_map.get(abs_norm)
        if image_id is None:
            continue

        img_pil = Image.open(abs_norm).convert("RGB")
        preds, confs = ssd_infer_one(
            model, device, img_pil,
            conf_thr=args.conf_thr, nms_iou=args.nms_iou,
            classes_keep=classes_keep, label_shift=1
        )

        # 读取 GT 并计算指标
        W, H = img_pil.size
        label_path = to_label_path(abs_norm, images_dir, labels_dir)
        gts = read_gt_boxes(label_path, W, H, keep_classes=classes_keep) if os.path.isfile(label_path) else []

        if len(gts) > 0:
            matches, un_p, un_g = greedy_match(preds, gts, args.match_iou)
            tp = len(matches)
            fp = len(un_p)
            fn = len(un_g)
            n_gt = len(gts)
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
        else:
            # 无标签：使用代理 F1（以置信度作为参考）
            if len(confs) > 0:
                mean_conf = float(np.mean(confs))
                top3_conf = float(np.mean(sorted(confs)[-3:])) if len(confs) >= 3 else mean_conf
            else:
                mean_conf = top3_conf = 0.0
            f1 = top3_conf if PROXY_F1_MODE == "top3_conf" else mean_conf
            tp = fp = fn = n_gt = 0

        row = (
            int(image_id),
            float(args.conf_thr),
            float(args.nms_iou),
            float(args.match_iou),
            int(tp),
            int(fp),
            int(fn),
            int(n_gt),
            float(f1),
            int(MODEL_ID),
        )
        batch_rows.append(row)
        n_done += 1

        if len(batch_rows) >= args.batch_commit:
            insert_batch(cur, eval_table, batch_rows)
            conn.commit()
            batch_rows.clear()
            gc.collect()

        if n_done % 500 == 0:
            dt = time.time() - t0
            print(f"[PROGRESS] {n_done} images processed, {n_done / max(1.0, dt):.1f} img/s")

    if batch_rows:
        insert_batch(cur, eval_table, batch_rows)
        conn.commit()

    cur.close()
    conn.close()
    print(f"[DONE] 共处理 {n_done} 张图，结果已写入表 `{eval_table}`。")


if __name__ == "__main__":
    main()
