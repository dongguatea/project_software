from ultralytics import YOLO

model = YOLO('yolov12n.pt')  #yolov12n.pt or yolov12n.yaml

# Train the model
results = model.train(
  data='data.yaml',    #yaml文件位置
  epochs=300,           #训练轮次
  batch=64,            #一次训练量
  imgsz=640,
  scale=0.3,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=0.2,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="0",
  workers=0,
  project="zoo"
)

# Evaluate model performance on the validation set
metrics = model.val()


