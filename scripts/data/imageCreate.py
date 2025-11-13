from PIL import Image
from pathlib import Path

def create_image(image_path,size):
  for num in range(0,100):
    img_path = image_path / f"{num:05d}.bmp"
    image = Image.new("RGB",size,(255,255,255))
    image.save(img_path)
def create_label(cls,image_path):
  if cls == 'manbo':
    for num in range(0,100):
      label_path = image_path / f"{num:05d}.txt"
      with open(label_path,'w') as f:
        f.write("0 0.5 0.5 0.5 0.5")
  elif cls == 'panmao':
    for num in range(0,100):
      label_path = image_path / f"{num:05d}.txt"
      with open(label_path,'w') as f:
        f.write("1 0.5 0.5 0.5 0.5")
def main():
  size = []
  # 创建图片
  # for i in range(1,100):
  #   cfg = str(i)
  #   image_path = Path(__file__).parent / 'images' / cfg
  #   image_path.mkdir(parents=True, exist_ok=True)
  #   if i % 90 < 30:
  #     size.append((320, 256))
  #   elif i % 90 < 60:
  #     size.append((640, 512))
  #   else:
  #     size.append((1024, 1024))
    # create_image(image_path, size[-1]) 创建图片
    # print(f"Created image {cfg} with size {size[-1]}")
  rootdir = Path(__file__).parent
  imagedir = rootdir / 'images'

  temlabel = rootdir / 'template'
  for cls in temlabel.glob('*/'):
    for img_w in Path(temlabel / cls.name).glob('*/'):
      temTxtPath = temlabel / cls.name / img_w.name
      create_label(cls.name,temTxtPath)

if __name__ == '__main__':
  main()