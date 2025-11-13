import argparse
import os
import cv2
import numpy as np

def parse():
    parser = argparse.ArgumentParser(description="Detect objects using fixed gray thresholds or max-gray-delta, export YOLO.")
    parser.add_argument("--imagedir", default="E:\\data_manager\\pythonProject1\\yolov12\\脚本", help="图片所在文件夹路径")

    # 模式：two=两类固定阈值；maxdelta=最大灰度-减量（兼容老逻辑）
    parser.add_argument("--mode", choices=["two", "maxdelta"], default="two", help="检测模式")

    # 两类固定阈值模式参数
    parser.add_argument("--class-a", type=int, default=0, help="类别A(高亮)的类别号")
    parser.add_argument("--thr-high", type=int, default=215, help="类别A阈值：灰度>=thr_high")
    parser.add_argument("--class-b", type=int, default=1, help="类别B(次亮)的类别号")
    parser.add_argument("--thr-low", type=int, default=180, help="类别B阈值下界：thr_low<=灰度<thr_high")

    # 旧模式参数（保留兼容）
    parser.add_argument("--classnum", default=1, type=int, help="老模式的类别号（maxdelta模式时使用）")
    parser.add_argument("--delta", default=60, type=int, help="老模式：阈值=最大灰度-此值")

    parser.add_argument("--min-area", type=int, default=50, help="最小连通域面积（像素）")

    # 将输出视为目录（更符合YOLO标注组织）
    parser.add_argument("--txt-out", default="./labels", help="YOLO标注输出目录（按图片同名）")
    parser.add_argument("--vis-out", default="./vis", help="可视化输出目录（按图片同名）")

    # 形态学选项
    parser.add_argument("--morph", choices=["none", "open", "close"], default="none", help="形态学操作")
    parser.add_argument("--kernel", type=int, default=3, help="形态学核大小(奇数)")

    return parser.parse_args()

def ensure_gray(img):
    if img is None:
        raise FileNotFoundError("读取图片失败，请检查路径。")
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def compute_threshold(gray, delta):
    max_gray = int(gray.max())
    thr = int(np.floor(np.clip(max_gray - delta, 0, 255)))
    return max_gray, thr

def apply_morph(bin_img, morph, ksize):
    if morph == "none":
        return bin_img
    k = ksize if ksize % 2 == 1 else ksize + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if morph == "open":
        return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    else:  # "close"
        return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

def bboxes_from_binary(bin_img, min_area=50):
    # 用连通域统计直接拿 bbox，更稳：每个断开的目标一框
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    bboxes = []
    for lab in range(1, num):
        x, y, w, h, area = stats[lab]
        if area >= min_area:
            bboxes.append((int(x), int(y), int(w), int(h)))
    return bboxes

def to_yolo(x, y, w, h, img_w, img_h):
    x_c = (x + x + w) / 2.0
    y_c = (y + y + h) / 2.0
    return x_c / img_w, y_c / img_h, w / img_w, h / img_h

def draw_rects(vis_img, boxes, color, thickness=2):
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, thickness)
    return vis_img

def main():
    args = parse()

    os.makedirs(args.txt_out, exist_ok=True)
    os.makedirs(args.vis_out, exist_ok=True)

    image_files = [f for f in os.listdir(args.imagedir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff'))]

    for fname in image_files:
        img_path = os.path.join(args.imagedir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] 无法读取图片：{img_path}")
            continue

        # 可视化用BGR
        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.shape[2] == 4 else img.copy()

        gray = ensure_gray(img)
        H, W = gray.shape[:2]

        yolo_lines = []

        if args.mode == "two":
            # 类别A：>= thr_high
            _, bin_high = cv2.threshold(gray, args.thr_high, 255, cv2.THRESH_BINARY)
            bin_high = apply_morph(bin_high, args.morph, args.kernel)
            bboxes_high = bboxes_from_binary(bin_high, args.min_area)
            for (x, y, w, h) in bboxes_high:
                xc, yc, ww, hh = to_yolo(x, y, w, h, W, H)
                yolo_lines.append(f"{args.class-a if False else args.class_a} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            # 类别B：thr_low <= 灰度 < thr_high
            bin_low = cv2.inRange(gray, args.thr_low, max(0, args.thr_high - 1))
            bin_low = apply_morph(bin_low, args.morph, args.kernel)
            bboxes_low = bboxes_from_binary(bin_low, args.min_area)
            for (x, y, w, h) in bboxes_low:
                xc, yc, ww, hh = to_yolo(x, y, w, h, W, H)
                yolo_lines.append(f"{args.class-b if False else args.class_b} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

            # 可视化：两类不同颜色（A=绿色，B=蓝色）
            vis = draw_rects(vis, bboxes_high, (0, 255, 0), 2)
            vis = draw_rects(vis, bboxes_low, (255, 0, 0), 2)

            print(f"[INFO] {fname} | A(>= {args.thr_high})={len(bboxes_high)}  B([{args.thr_low},{args.thr_high}))={len(bboxes_low)}")

        else:  # maxdelta 模式（兼容旧逻辑）
            max_gray, thr = compute_threshold(gray, args.delta)
            _, bin_img = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
            bin_img = apply_morph(bin_img, args.morph, args.kernel)
            bboxes = bboxes_from_binary(bin_img, args.min_area)
            for (x, y, w, h) in bboxes:
                xc, yc, ww, hh = to_yolo(x, y, w, h, W, H)
                yolo_lines.append(f"{args.classnum} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            vis = draw_rects(vis, bboxes, (0, 255, 0), 2)
            print(f"[INFO] {fname} | maxdelta: max={int(gray.max())} thr={thr}  n={len(bboxes)}")

        # 写 YOLO txt（与图片同名）
        txt_name = os.path.splitext(fname)[0] + ".txt"
        txt_path = os.path.join(args.txt_out, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in yolo_lines:
                f.write(line + "\n")
        # 保存可视化
        vis_path = os.path.join(args.vis_out, fname)
        cv2.imwrite(vis_path, vis)

    print("[INFO] 文件夹处理完成。")

if __name__ == "__main__":
    main()
