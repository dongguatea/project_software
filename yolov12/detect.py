"""
img_path
  └─► to_label_path ──► label_path
                         ├─► count_gt ──► n_gt
                         └─► read_gt_boxes (xywhn_to_xyxy 内部被调用)
                                 ▲
预测结果（Ultralytics） → preds ──┼──► greedy_match (内部用 box_iou)
                                 └─► 得到 TP / FP / FN → 算 F1

推理结果result包含的内容：
for r in results:
    r.path         # 这张图的路径（str）
    r.orig_shape   # 原图尺寸 (h, w)
    r.boxes        # Boxes 对象（可能为 None 或长度为该图检测数）
    # 取出框：
    for b in r.boxes:
        cls_id = int(b.cls[0])      # 类别ID (tensor -> python int)
        conf   = float(b.conf[0])   # 置信度
        x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]  # 像素坐标

"""


import os, glob, re
from pathlib import Path
import mysql.connector
import numpy as np
import gc
from ultralytics import YOLO

# ===== 基本配置 =====
MODEL_PATH = "best.pt"
IMAGES_DIR = r"F:/data_manager/pythonProject1/data/images"    # 图像主文件夹
LABELS_DIR = r"F:/data_manager/pythonProject1/data/labels"    # 标签主文件夹


DB_Config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'asd515359',
    'database': 'config_db',
    'port' : 3306
}

# 统一口径（务必所有批次一致）,定义进行图像检测时的参数
IMG_SIZE   = 640
CONF_THR   = 0.3
NMS_IOU    = 0.6
MATCH_IOU  = 0.50
AUGMENT    = False
DEVICE     = 0          # 或 'cpu'
CLASSES_KEEP = None     # eg. [0,1]；None 表示全部类别

BATCH_COMMIT = 5000

PROXY_F1_MODE = 'mean_conf'

model = YOLO('yolov12n.pt')
# ---------- 工具函数 ----------
"""把任意文件绝对路径 -> 规范化的 image_path 键：'images/...'（统一正斜杠）"""
def norm_path_to_key(p: Path) -> str:
    rel = p.resolve().relative_to(Path(IMAGES_DIR).resolve())
    key = str(Path(IMAGES_DIR) / rel)
    return key.replace("\\", "/")

"""将images转换成labels代表标签位置"""
def to_label_path_from_key(image_key: str) -> str:
    """'images/cls/cfg/file.jpg' -> 'labels/cls/cfg/file.txt'"""
    rel = Path(image_key).relative_to(Path(IMAGES_DIR).resolve())
    return f"{LABELS_DIR}/{Path(rel).with_suffix('.txt')}".replace("\\", "/")

#计算实际存在的gt值
def count_gt(label_path: str, keep_classes=None) -> int:
    if not os.path.isfile(label_path):
        return 0
    n = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            #获取每个标注文件中的类型：cid,如果cid在需要识别的类中，则gt数+1
            if len(parts) != 5:
                continue
            cid = int(float(parts[0]))
            if (keep_classes is None) or (cid in keep_classes):
                n += 1
    return n

def xywhn_to_xyxy(xywhn, img_w, img_h):
    x, y, w, h = xywhn
    return [(x - w/2) * img_w, (y - h/2) * img_h, (x + w/2) * img_w, (y + h/2) * img_h]
def read_gt_boxes(label_path: str, img_w: int, img_h: int, keep_classes=None):
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
            gts.append({'cls': cls_id, 'box': xywhn_to_xyxy([x,y,w,h], img_w, img_h)})
    return gts

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    b_area = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = a_area + b_area - inter
    return inter/union if union > 0 else 0.0

def greedy_match(preds, gts, iou_thr):
    """同类、IoU>=阈值，贪心一对一匹配"""
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if p['cls'] != g['cls']:
                continue
            iou = box_iou(p['box'], g['box'])
            if iou >= iou_thr:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    matched_p, matched_g, matches = set(), set(), []
    for iou, pi, gi in pairs:
        if pi not in matched_p and gi not in matched_g:
            matched_p.add(pi); matched_g.add(gi)
            matches.append((pi, gi, iou))
    un_p = [i for i in range(len(preds)) if i not in matched_p]
    un_g = [i for i in range(len(gts))   if i not in matched_g]
    return matches, un_p, un_g

def natural_sort_key(path_str: str):
    p = Path(path_str)
    # 统一为相对 images 的路径 key：images/cls/ConfigID/file.ext
    try:
        rel = p.resolve().relative_to(Path(IMAGES_DIR).resolve())
    except Exception:
        rel = p  # 如果不在 IMAGES_DIR 下，就直接用 p
    parts = rel.parts  # ('manbo','123','00211.bmp') 期望长度>=3
    cls  = parts[0] if len(parts) > 0 else ""
    cfg  = parts[1] if len(parts) > 1 else ""
    fname= parts[2] if len(parts) > 2 else p.name
    stem = Path(fname).stem

    # 1) 类别顺序（可自定义优先级：ddg在前，cvn在后）
    cls_order = {"manbo": 0, "panmao": 1}  # 不在映射内的类排最后
    cls_key = (cls_order.get(cls, 99), cls.lower())

    # 2) ConfigID：数字优先；非纯数字则提取首个数字，否则按字符串兜底
    if cfg.isdigit():
        cfg_key = (0, int(cfg), cfg.lower())
    else:
        m = re.search(r"\d+", cfg)
        cfg_key = (1, int(m.group()) if m else 0, cfg.lower())

    # 3) 文件编号：取文件名中的第一段数字并数值比较；没有数字就按字母
    m2 = re.search(r"\d+", stem)
    if m2:
        file_key = (0, int(m2.group()), stem.lower())
    else:
        file_key = (1, 0, stem.lower())

    # 最后再附带完整小写路径，确保全局稳定
    return (cls_key, cfg_key, file_key, str(rel).lower())

def detect_one_picture(img_path,img_id):
    usemodel = 0
    res_list = model.predict(
        source=img_path, imgsz=IMG_SIZE, conf=CONF_THR, iou=NMS_IOU,
        augment=AUGMENT, device=DEVICE, save=False, stream=False, workers=0
    )
    r = res_list[0]
    try:
        H, W = r.orig_shape
        # 预测框 & 置信度
        preds, confs = [], []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                if (CLASSES_KEEP is not None) and (cls_id not in CLASSES_KEEP):
                    continue
                x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]
                conf = float(b.conf[0])
                preds.append({'cls': cls_id, 'box': [x1,y1,x2,y2]})
                confs.append(conf)
        n_pred = len(preds)

        # 标签路径
        label_path = to_label_path_from_key(img_path)
        gts = read_gt_boxes(label_path, W, H, keep_classes=CLASSES_KEEP) if os.path.isfile(label_path) else []

        if len(gts) > 0:
            matches, un_p, un_g = greedy_match(preds, gts, MATCH_IOU)
            tp = len(matches); fp = len(un_p); fn = len(un_g)
            n_gt = len(gts)
            denom = 2*tp + fp + fn
            f1 = (2*tp / denom) if denom > 0 else 0.0
        else:
            # 无标注：f1 用代理
            if n_pred > 0:
                mean_conf = float(np.mean(confs))
                top3_conf = float(np.mean(sorted(confs)[-3:])) if n_pred >= 3 else mean_conf
            else:
                mean_conf = top3_conf = 0.0
            f1 = top3_conf if PROXY_F1_MODE == 'top3_conf' else mean_conf
            tp = fp = fn = n_gt = 0

        return (img_id, usemodel, CONF_THR, NMS_IOU, MATCH_IOU, tp, fp, fn,n_pred, n_gt, float(f1))
    finally:
        del res_list
        del r
        gc.collect()
        
sql = """
INSERT INTO eval_metrics
VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""

def normkey(p: str) -> str:
    return str(Path(p).resolve()).replace("\\", "/").lower()

def main():
    # 1) 收集全部图片文件
    img_files = []

    for ext in ("png","bmp"):
        img_files.extend(Path(IMAGES_DIR).rglob(f"*.{ext}"))
    img_files = [str(Path(p)) for p in img_files]
    img_files.sort(key=natural_sort_key)
    if not img_files:
        raise FileNotFoundError("未在 images/{manbo|panmao}/{ConfigID} 下找到图片")

    # 把绝对路径 -> image_path key 的映射先建好（后面用 r.path 快速取）
    fs_abs2key = {}
    count = 0
    for p in img_files:
        count += 1
        p = normkey(p)
        fs_abs2key[p] = count

    # 2) 预检：images 表必须覆盖所有 image_path
    conn = mysql.connector.connect(**DB_Config)
    cur  = conn.cursor()
    cur.execute("SELECT image_id, image_path FROM images")
    rows = cur.fetchall()
    key2id = {str(row[1]).replace("\\","/").lower() : int(row[0]) for row in rows} #image_path : image_id

    db_keys = set(k for k in key2id.keys())
    needed_keys = set(k for k in fs_abs2key.keys())

    missing = sorted(needed_keys - db_keys)
    if missing:
        # 给出缺失清单并中止（避免推理后写不了库）
        with open("missing_image_paths.txt", "w", encoding="utf-8") as f:
            for k in missing: f.write(k + "\n")
        cur.close(); conn.close()
        raise RuntimeError(
            f"images 表缺少 {len(missing)} 条 image_path。"
            f"已写出 missing_image_paths.txt，请先把这些路径补充进 images.image_path（绝对路径，前缀 'images/'）。"
        )

    # 3) 推理（流式）
    
    batch = []
    for image_key in fs_abs2key.keys():
        # image_key = norm_path_to_key(image_key)# 直接用 'images/cls/cfg/file.jpg'
        image_id = key2id.get(image_key)
        if image_id is None:
            continue
        row = detect_one_picture(image_key, image_id)
        batch.append(row)

        if len(batch) >= BATCH_COMMIT:
            cur.executemany(sql, batch)
            conn.commit()
            batch.clear()
            gc.collect()

    if batch:
        cur.executemany(sql, batch)
        conn.commit()

    cur.close(); conn.close()
    print("完成：全部图片推理并写入 eval_metrics。")

if __name__ == "__main__":
    main()
