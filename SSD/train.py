import argparse
import os
#修改，新增torchcache变量，让代码在这条路径中找到对应的权重文件
os.environ["TORCH_HOME"] = r"E:\data_manager\torch_cache"  # 原始字符串避免反斜杠转义
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import torch,torch.hub
torch.hub.set_dir(os.path.join(os.environ["TORCH_HOME"], "hub"))
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models import VGG16_Weights
from torchvision.transforms.functional import to_tensor
from torchvision.ops import box_iou

from PIL import Image
from tqdm import tqdm


resume_path = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 强制使用CPU
# =========================
# 1) 常量与工具
# =========================
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    images, targets = zip(*batch)  # list[Tensors], list[Dict]
    return list(images), list(targets)

def guess_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2) 数据集
# =========================
class VOCDataset(Dataset):
    """
    返回目标检测统一格式：
      target = {"boxes": FloatTensor[N,4], "labels": Int64Tensor[N], "image_id": Int64Tensor[1], "area": FloatTensor[N], "iscrowd": Int64Tensor[N]}
    """
    def __init__(self, voc_root: str, year: str = "2007", image_set: str = "trainval", classes=None):
        super().__init__()
        self.voc_root = Path(voc_root)
        self.year = year
        self.image_set = image_set
        self.root = self.voc_root / f"VOC{year}"
        assert self.root.exists(), f"VOC{year} not found under {voc_root}"

        split_file = self.root / "ImageSets" / "Main" / f"{image_set}.txt"
        assert split_file.exists(), f"Split file not found: {split_file}"
        with open(split_file, "r") as f:
            self.ids = [x.strip() for x in f.readlines() if x.strip()]

        self.img_dir = self.root / "JPEGImages"
        self.ann_dir = self.root / "Annotations"

        self.class_names = classes if (classes and len(classes) > 0) else VOC_CLASSES
        self.class_to_idx = {name: i + 1 for i, name in enumerate(self.class_names)}  # 背景0，前景从1

    def __len__(self):
        return len(self.ids)

    def _parse_annotation(self, ann_path: Path):
        tree = ET.parse(ann_path.open("rb"))
        root = tree.getroot()
        boxes, labels, iscrowd, areas = [], [], [], []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name not in self.class_to_idx:
                continue
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text); ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text); ymax = float(bnd.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])
            iscrowd.append(0)
            areas.append(max(0.0, xmax - xmin) * max(0.0, ymax - ymin))
        import torch
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        }

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        img_path = self.img_dir / f"{img_id}.jpg"
        ann_path = self.ann_dir / f"{img_id}.xml"
        img = Image.open(img_path).convert("RGB")
        image_tensor = to_tensor(img)
        target = self._parse_annotation(ann_path)
        target["image_id"] = torch.tensor([index], dtype=torch.int64)
        return image_tensor, target

class YOLODataset(Dataset):
    """
    YOLO txt：每行 class cx cy w h（归一化）。支持子目录与 .bmp。
    images_dir 与 labels_dir 目录结构需镜像；list_file 可指定相对 images_dir 的图片清单。
    """
    def __init__(self, images_dir: str, labels_dir: str, classes_txt: str, list_file: str = ""):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        assert self.images_dir.exists() and self.labels_dir.exists()
        assert classes_txt and os.path.isfile(classes_txt), "classes_txt is required"
        with open(classes_txt, "r", encoding="utf-8") as f:
            self.class_names = [x.strip() for x in f if x.strip()]
        self.class_to_idx = {i: i+1 for i in range(len(self.class_names))}  # 模型标签：1..K

        if list_file and os.path.isfile(list_file):
            with open(list_file, "r", encoding="utf-8") as f:
                paths = [x.strip() for x in f if x.strip()]
            self.img_paths = [Path(p) if os.path.isabs(p) else (self.images_dir / p) for p in paths]
        else:
            self.img_paths = []
            for e in ("*.jpg","*.jpeg","*.png","*.bmp"):
                self.img_paths.extend(sorted(self.images_dir.rglob(e)))
        assert len(self.img_paths) > 0, f"No images found in {self.images_dir}"
        
    def __len__(self): return len(self.img_paths)

    def __getitem__(self, index: int):
        img_path = self.img_paths[index]
        rel = img_path.relative_to(self.images_dir)
        lbl_path = (self.labels_dir / rel).with_suffix(".txt")
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        boxes, labels, iscrowd, areas = [], [], [], []
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    ps = line.strip().split()
                    if len(ps) != 5: continue
                    cls, cx, cy, w, h = ps
                    cls = int(float(cls)); cx, cy, w, h = map(float, (cx, cy, w, h))
                    x1 = (cx - w/2.0) * W; y1 = (cy - h/2.0) * H
                    x2 = (cx + w/2.0) * W; y2 = (cy + h/2.0) * H
                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.class_to_idx.get(cls, 0))
                    iscrowd.append(0)
                    areas.append(max(0.0, x2-x1) * max(0.0, y2-y1))
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return to_tensor(img), target

# =========================
# 3) 模型
# =========================
def build_ssd_model(num_classes: int):
    return ssd300_vgg16(
        weights=None,  # 不加载COCO检测头
        weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,  # 加载VGG16特征权重
        num_classes=num_classes
    )

# =========================
# 4) mAP 评估（VOC07 AP 和 COCO风格 AP@[.5:.95]）
# =========================
@torch.no_grad()
def evaluate_map(model, data_loader, class_names: List[str], device,
                 iou_thresholds: List[float] = None, score_thr: float = 0.0) -> Dict[str, float]:
    """
    返回：
      {"mAP_50": ..., "mAP": ..., "AP50_per_class": {...}, "AP_per_class": {...}}
    """
    model.eval()
    if iou_thresholds is None:
        iou_thresholds = [0.5] + [0.55 + 0.05*i for i in range(8)] + [0.95]  # 0.5:0.95 步长0.05

    # 累积 GT & Pred
    # per class:
    #   gts[class][img_id] = boxes tensor (M,4)
    #   preds[class] = [(img_id, score, box tensor)]
    from collections import defaultdict
    gts = {c: defaultdict(list) for c in range(1, 1+len(class_names))}
    preds = {c: [] for c in range(1, 1+len(class_names))}
    n_gt_per_class = {c: 0 for c in range(1, 1+len(class_names))}

    for images, targets in tqdm(data_loader, desc="Eval", ncols=100):
        images = [img.to(device) for img in images]
        outputs = model(images)  # list of dicts with boxes/labels/scores
        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())
            # 收集GT
            gt_boxes = tgt["boxes"].to("cpu")
            gt_labels = tgt["labels"].to("cpu")
            for c in range(1, 1+len(class_names)):
                cls_mask = (gt_labels == c)
                if cls_mask.any():
                    gts[c][img_id].append(gt_boxes[cls_mask])
                    n_gt_per_class[c] += int(cls_mask.sum().item())
            # 收集Pred
            boxes = out["boxes"].detach().to("cpu")
            labels = out["labels"].detach().to("cpu")
            scores = out["scores"].detach().to("cpu")
            if score_thr > 0:
                keep = scores >= score_thr
                boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
            for b, l, s in zip(boxes, labels, scores):
                if int(l) in preds:
                    preds[int(l)].append((img_id, float(s.item()), b))

    # 将 list -> tensor 并合并同一图像的 GT
    for c in gts:
        for img_id in list(gts[c].keys()):
            if len(gts[c][img_id]) > 0:
                gts[c][img_id] = torch.vstack(gts[c][img_id])
            else:
                gts[c][img_id] = torch.zeros((0,4))

    def ap_at_iou(c: int, thr: float) -> float:
        """计算某类别在单个 IoU 阈值下的 AP（插值积分）。"""
        if n_gt_per_class[c] == 0:  # 该类没有 GT
            return float("nan")
        # 该类的预测按 score 降序
        pc = sorted(preds[c], key=lambda x: x[1], reverse=True)
        tp = torch.zeros(len(pc))
        fp = torch.zeros(len(pc))
        # 针对每个图像记录已匹配的 gt 索引
        matched = {}
        for i, (img_id, score, pbox) in enumerate(pc):
            gt_boxes = gts[c].get(img_id, torch.zeros((0,4)))
            if gt_boxes.numel() == 0:
                fp[i] = 1
                continue
            if img_id not in matched:
                matched[img_id] = torch.zeros(gt_boxes.size(0), dtype=torch.bool)
            ious = box_iou(pbox.view(1,4), gt_boxes).squeeze(0)  # (M,)
            max_iou, max_idx = (ious.max().item(), int(ious.argmax().item())) if gt_boxes.size(0) > 0 else (0.0, -1)
            if max_iou >= thr and not matched[img_id][max_idx]:
                tp[i] = 1
                matched[img_id][max_idx] = True
            else:
                fp[i] = 1
        # 累积
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recalls = tp_cum / max(1, n_gt_per_class[c])
        precisions = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)
        # 插值精度包络
        mrec = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
        mpre = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
        for i in range(mpre.size(0)-1, 0, -1):
            mpre[i-1] = torch.maximum(mpre[i-1], mpre[i])
        # 计算面积（VOC10+ 通用做法；非11点）
        idx = (mrec[1:] != mrec[:-1]).nonzero().squeeze(1)
        ap = float(torch.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]).item())
        return ap

    AP50_per_class = {}
    AP_per_class_mean = {}
    aps_50 = []
    aps_5095 = []
    for c in range(1, 1+len(class_names)):
        ap_50 = ap_at_iou(c, 0.5)
        AP50_per_class[class_names[c-1]] = 0.0 if ap_50 != ap_50 else ap_50
        aps_50.append(AP50_per_class[class_names[c-1]])
        # 0.5:0.95
        ap_list = []
        for thr in iou_thresholds:
            ap_t = ap_at_iou(c, thr)
            ap_list.append(0.0 if ap_t != ap_t else ap_t)
        AP_per_class_mean[class_names[c-1]] = sum(ap_list) / len(ap_list) if len(ap_list) else 0.0
        aps_5095.append(AP_per_class_mean[class_names[c-1]])

    metrics = {
        "mAP_50": sum(aps_50) / max(1, len(aps_50)),
        "mAP": sum(aps_5095) / max(1, len(aps_5095)),
        "AP50_per_class": AP50_per_class,
        "AP_per_class": AP_per_class_mean
    }
    return metrics

# =========================
# 5) 训练（VOC / YOLO） + 验证 + 早停
# =========================
def train_loop(model, optimizer, lr_scheduler, train_loader, device):
    model.train()
    running = 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    for images, targets in tqdm(train_loader, desc="Train", ncols=100):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()
        running += float(loss.item())
    lr_scheduler.step()
    return running / max(1, len(train_loader))

def save_ckpt(path: Path, model, classes, epoch: int):
    torch.save({"model": model.state_dict(), "classes": classes, "epoch": epoch}, path)

def early_stopping_update(best_metric: float, current: float, patience: int, counter: int, min_delta: float):
    improved = (current - best_metric) > min_delta
    if improved:
        return current, 0, True
    else:
        counter += 1
        return best_metric, counter, False

def train_ssd_voc(
    voc_root: str,
    year: str,
    image_set_train: str,
    image_set_val: str,
    out_path: str,
    epochs: int = 24,
    batch_size: int = 8,
    lr: float = 0.002,
    num_workers: int = 4,
    classes_txt: str = "",
    eval_interval: int = 1,
    metric: str = "map50",         # "map50" 或 "map"
    patience: int = 10,
    min_delta: float = 0.0
):
    set_seed(2024)
    device = torch.device(guess_device())
    print(f"[INFO] Device: {device}")

    custom_classes = []
    if classes_txt and os.path.isfile(classes_txt):
        with open(classes_txt, "r", encoding="utf-8") as f:
            custom_classes = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Using custom classes: {custom_classes}")

    ds_tr = VOCDataset(voc_root, year, image_set_train, classes=custom_classes)
    ds_va = VOCDataset(voc_root, year, image_set_val, classes=ds_tr.class_names)
    num_classes = 1 + len(ds_tr.class_names)

    model = build_ssd_model(num_classes=num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*epochs), int(0.85*epochs)], gamma=0.1)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                       collate_fn=collate_fn, pin_memory=(device.type=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                       collate_fn=collate_fn, pin_memory=(device.type=="cuda"))

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    best_metric = -1e9; counter = 0
    choose = "mAP_50" if metric.lower()=="map50" else "mAP"

    for epoch in range(1, epochs+1):
        t0 = time.time()
        tr_loss = train_loop(model, optimizer, lr_scheduler, dl_tr, device)
        msg = f"[INFO] Epoch {epoch}/{epochs}  train_loss={tr_loss:.4f}"

        if (epoch % eval_interval) == 0:
            metrics = evaluate_map(model, dl_va, ds_tr.class_names, device)
            cur = metrics[choose]
            msg += f"  |  val_{choose}={cur:.4f}  (mAP@0.5={metrics['mAP_50']:.4f}, mAP@0.5:0.95={metrics['mAP']:.4f})"
            # 保存最新
            save_ckpt(out_path, model, ds_tr.class_names, epoch)
            # 保存最优 + 早停
            best_metric, counter, improved = early_stopping_update(best_metric, cur, patience, counter, min_delta)
            if improved:
                save_ckpt(out_path.with_suffix(".best.pth"), model, ds_tr.class_names, epoch)
                print(f"[INFO] New best {choose}={best_metric:.4f} @ epoch {epoch}. Saved: {out_path.with_suffix('.best.pth')}")
            elif counter >= patience:
                print(f"[EARLY STOP] No improvement in {patience} eval steps. Stop at epoch {epoch}.")
                break

        dt = time.time() - t0
        print(msg + f"  |  time={dt:.1f}s")

    print(f"[INFO] Training finished. Last weights: {out_path}")

def train_ssd_yolo(
    images_dir: str,
    labels_dir: str,
    out_path: str,
    classes_txt: str,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 0.001,
    num_workers: int = 4,
    list_file: str = "train.txt",            # 训练清单（相对 images_dir）
    val_list_file: str = "val.txt",        # 验证清单（相对 images_dir）
    eval_interval: int = 1,
    metric: str = "map50",          # "map50" 或 "map"
    patience: int = 10,
    min_delta: float = 0.0
):
    set_seed(2024)
    # device = torch.device(guess_device())
    device = torch.device("cpu")

    ds_tr = YOLODataset(images_dir, labels_dir, classes_txt=classes_txt, list_file=list_file)
    # 验证集：优先使用 val_list_file；否则抽取 10% 作为验证
    if val_list_file and os.path.isfile(val_list_file):
        ds_va = YOLODataset(images_dir, labels_dir, classes_txt=classes_txt, list_file=val_list_file)
    else:
        # 简易划分（不打乱目录结构）
        n = len(ds_tr)
        idx = torch.randperm(n)
        k = max(1, int(0.1 * n))
        subset_va = [ds_tr.img_paths[i] for i in idx[:k]]
        subset_tr = [ds_tr.img_paths[i] for i in idx[k:]]
        # 写临时清单
        tmp_dir = Path(".splits_tmp"); tmp_dir.mkdir(exist_ok=True)
        (tmp_dir/"train.txt").write_text("\n".join([p.relative_to(ds_tr.images_dir).as_posix() for p in subset_tr]), encoding="utf-8")
        (tmp_dir/"val.txt").write_text("\n".join([p.relative_to(ds_tr.images_dir).as_posix() for p in subset_va]), encoding="utf-8")
        ds_tr = YOLODataset(images_dir, labels_dir, classes_txt=classes_txt, list_file=str(tmp_dir/"train.txt"))
        ds_va = YOLODataset(images_dir, labels_dir, classes_txt=classes_txt, list_file=str(tmp_dir/"val.txt"))

    num_classes = 1 + len(ds_tr.class_names)

    model = build_ssd_model(num_classes=num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*epochs), int(0.85*epochs)], gamma=0.1)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                       collate_fn=collate_fn, pin_memory=(device.type=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                       collate_fn=collate_fn, pin_memory=(device.type=="cuda"))

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    best_metric = -1e9; counter = 0
    choose = "mAP_50" if metric.lower()=="map50" else "mAP"

    for epoch in range(1, epochs+1):
        t0 = time.time()
        tr_loss = train_loop(model, optimizer, lr_scheduler, dl_tr, device)
        msg = f"[INFO] Epoch {epoch}/{epochs}  train_loss={tr_loss:.4f}"

        if (epoch % eval_interval) == 0:
            metrics = evaluate_map(model, dl_va, ds_tr.class_names, device)
            cur = metrics[choose]
            msg += f"  |  val_{choose}={cur:.4f}  (mAP@0.5={metrics['mAP_50']:.4f}, mAP@0.5:0.95={metrics['mAP']:.4f})"
            save_ckpt(out_path, model, ds_tr.class_names, epoch)
            best_metric, counter, improved = early_stopping_update(best_metric, cur, patience, counter, min_delta)
            if improved:
                save_ckpt(out_path.with_suffix(".best.pth"), model, ds_tr.class_names, epoch)
                print(f"[INFO] New best {choose}={best_metric:.4f} @ epoch {epoch}. Saved: {out_path.with_suffix('.best.pth')}")
            elif counter >= patience:
                print(f"[EARLY STOP] No improvement in {patience} eval steps. Stop at epoch {epoch}.")
                break

        dt = time.time() - t0
        print(msg + f"  |  time={dt:.1f}s")

    print(f"[INFO] Training finished. Last weights: {out_path}")

# =========================
# 6) CLI（已移除 detect 子命令）
# =========================
def main():
    parser = argparse.ArgumentParser(description="SSD300 Training with mAP eval & Early Stopping")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # VOC 训练
    p_voc = subparsers.add_parser("train_voc", help="Train SSD on VOC with val & early stopping")
    p_voc.add_argument("--voc_root", type=str, required=True)
    p_voc.add_argument("--year", type=str, default="2007", choices=["2007","2012"])
    p_voc.add_argument("--image_set_train", type=str, default="trainval")
    p_voc.add_argument("--image_set_val", type=str, default="test")  # VOC2007: 没有val，默认test
    p_voc.add_argument("--epochs", type=int, default=24)
    p_voc.add_argument("--batch_size", type=int, default=8)
    p_voc.add_argument("--lr", type=float, default=0.002)
    p_voc.add_argument("--num_workers", type=int, default=4)
    p_voc.add_argument("--out", type=str, default="weights/ssd_voc.pth")
    p_voc.add_argument("--classes_txt", type=str, default="")
    p_voc.add_argument("--eval_interval", type=int, default=1)
    p_voc.add_argument("--metric", type=str, default="map50", choices=["map50","map"])
    p_voc.add_argument("--patience", type=int, default=10)
    p_voc.add_argument("--min_delta", type=float, default=0.0)

    # YOLO 训练
    p_yolo = subparsers.add_parser("train_yolo", help="Train SSD on YOLO-format dataset with val & early stopping")
    #修改：直接将images_dir等设置为已知
    p_yolo.add_argument("--images_dir", type=str, default=r'E:\data_manager\pythonProject1\data\images')
    p_yolo.add_argument("--labels_dir", type=str, default=r'E:\data_manager\pythonProject1\data\labels')
    p_yolo.add_argument("--classes_txt", type=str, default=r'E:\data_manager\pythonProject1\data\class.txt')
    p_yolo.add_argument("--epochs", type=int, default=50)
    p_yolo.add_argument("--batch_size", type=int, default=8)
    p_yolo.add_argument("--lr", type=float, default=0.001)
    p_yolo.add_argument("--num_workers", type=int, default=0)
    p_yolo.add_argument("--out", type=str, default="weights/ssd_yolo.pth") #输出的权重路径
    # 修改：直接用相对路径会导致查找list_file出现问题
    p_yolo.add_argument("--list_file", type=str, default=r"E:\data_manager\pythonProject1\data\train.txt")
    p_yolo.add_argument("--val_list_file", type=str, default=r"E:\data_manager\pythonProject1\data\val.txt")
    p_yolo.add_argument("--eval_interval", type=int, default=1)
    p_yolo.add_argument("--metric", type=str, default="map50", choices=["map50","map"])
    p_yolo.add_argument("--patience", type=int, default=10)
    p_yolo.add_argument("--min_delta", type=float, default=0.0)

    args = parser.parse_args()

    if args.cmd == "train_voc":
        train_ssd_voc(
            voc_root=args.voc_root,
            year=args.year,
            image_set_train=args.image_set_train,
            image_set_val=args.image_set_val,
            out_path=args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            classes_txt=args.classes_txt,
            eval_interval=args.eval_interval,
            metric=args.metric,
            patience=args.patience,
            min_delta=args.min_delta
        )
    elif args.cmd == "train_yolo":
    # 修改：增加device和num_classes的定义
        # device = torch.device(guess_device())
        device = torch.device("cpu")
        num_classes = 1  # + 前景类
        if os.path.isfile(args.classes_txt):
            with open(args.classes_txt, "r", encoding="utf-8") as f:
                num_classes += len([line.strip() for line in f if line.strip()])
        model = build_ssd_model(num_classes=num_classes).to(device)

        if resume_path and os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"[INFO] resumed from {resume_path}, epoch {ckpt.get('epoch')}")
            if resume_path:
                ckpt = torch.load(resume_path, map_location=device)
                model.load_state_dict(ckpt["model"])
                print(f"[INFO] resumed from {resume_path}, epoch {ckpt.get('epoch')}")
        train_ssd_yolo(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            out_path=args.out,
            classes_txt=args.classes_txt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            list_file=args.list_file,
            val_list_file=args.val_list_file,
            eval_interval=args.eval_interval,
            metric=args.metric,
            patience=args.patience,
            min_delta=args.min_delta
        )

if __name__ == "__main__":
    main()
