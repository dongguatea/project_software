# train.py  —— 高精度配置（RTX 3090 24 GB）

from ultralytics import YOLO

model = YOLO('')        # or 'yolov12l.pt' / 'yolov12l.yaml'

# ② 训练
results = model.train(
    data='',            # 数据集 YAML
    epochs=300,                         # 充足训练轮次 + 早停
    imgsz=1024,                         # 输入分辨率（Letterbox 后大小）
    batch=8,                            # 物理 batch；24 GB + AMP 可安全运行
    accumulate=8,                       # 梯度累积 ⇒ 等效全局 batch = 64
    # amp=True,                           # 混合精度，省显存 & 提速
    # rect=True,                          # 矩形批次，减少填充
    patience=50,                        # 早停策略
    optimizer='SGD',                    # 大数据常用 SGD + momentum
    workers=8,                          # DataLoader 线程
    mosaic=1.0,                         # ✱ 高级增强保持开启
    mixup=0.15,
    copy_paste=0.4,
    project='IR',
    name='yolov12l_1024_acc64',         # 子目录便于多实验对比
    device=0                            # 单卡
)

# ③ 评估（加 --rect & imgsz 保持一致；可选 --tta 提升 0.5~1 mAP）
metrics = model.val(imgsz=1024, rect=True, amp=True)
