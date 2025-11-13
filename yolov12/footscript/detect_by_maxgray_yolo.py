import argparse
import os
import cv2
import numpy as np
def parse():
    parser = argparse.ArgumentParser(description="Detect objects using YOLO and max gray value method.")
    parser.add_argument("--imagedir",default="E:\\data_manager\\pythonProject1\\yolov12\\脚本",help="图片所在文件夹路径")
    parser.add_argument("--classnum",default=1,type=int,help="类别")
    parser.add_argument("--delta",default=60,type=int,help="灰度值阈值")
    parser.add_argument("--min-area", type=int, default=50, help="最小连通域面积（像素）")
    parser.add_argument("--txt-out", default="labels.txt", help="YOLO标注输出路径")
    parser.add_argument("--vis-out", default="vis.jpg", help="带框可视化输出路径")
    parser.add_argument("--close-k", type=int, default=3, help="闭运算核大小(奇数)，0不做")
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

def find_components(gray, thr, min_area=50, close_k=3):
    _, bin_img = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    if close_k and close_k > 0:
        k = close_k if close_k % 2 == 1 else close_k + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    return contours, bin_img

def bbox_from_contour(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h  # 左上角与宽高（像素）

def to_yolo(x, y, w, h, img_w, img_h):
    # YOLO: 中心点与宽高，且归一化
    x_c = (x + x + w) / 2.0
    y_c = (y + y + h) / 2.0
    return x_c / img_w, y_c / img_h, w / img_w, h / img_h

def draw_rects(vis_img, boxes, color=(0, 255, 0), thickness=2):
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, thickness)
    return vis_img

def main():
    args = parse()

    # 列出文件夹下所有图片文件
    image_files = [f for f in os.listdir(args.imagedir) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

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

        max_gray, thr = compute_threshold(gray, args.delta)
        print(f"[INFO] {fname} 最大灰度: {max_gray} | 阈值: {thr}")

        contours, _ = find_components(gray, thr, args.min_area, args.close_k)
        print(f"[INFO] 检测到高亮目标数: {len(contours)}")

        bboxes = [bbox_from_contour(c) for c in contours]
        yolo_lines = []
        for (x, y, w, h) in bboxes:
            x_c_n, y_c_n, w_n, h_n = to_yolo(x, y, w, h, W, H)
            yolo_lines.append(f"{args.classnum} {x_c_n:.6f} {y_c_n:.6f} {w_n:.6f} {h_n:.6f}")

        # 为每张图片单独保存 txt（同名不同后缀）
        txt_name = os.path.splitext(fname)[0] + ".txt"
        txt_path = os.path.join(args.txt_out, txt_name)
        os.makedirs(args.txt_out, exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in yolo_lines:
                f.write(line + "\n")
        print(f"[INFO] 已写出YOLO标注：{txt_path}")

        # 保存可视化结果
        vis = draw_rects(vis, bboxes, color=(0, 255, 0), thickness=2)
        os.makedirs(args.vis_out, exist_ok=True)
        vis_path = os.path.join(args.vis_out, fname)
        cv2.imwrite(vis_path, vis)
        print(f"[INFO] 已保存可视化：{vis_path}")

    print("[INFO] 文件夹处理完成。")
if __name__ == "__main__":
    main()