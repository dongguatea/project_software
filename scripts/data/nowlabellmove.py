from pathlib import Path
import shutil
from PIL import Image
import mysql.connector
import configparser
import os
from typing import Dict

def load_db_config(config_path: str = None) -> Dict[str, any]:
    """从config.ini文件加载数据库配置"""
    if config_path is None:
        # 默认在database目录下查找config.ini
        config_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'config.ini')
    
    if not os.path.exists(config_path):
        print(f"警告: 配置文件不存在 {config_path}，使用默认配置")
        return {
            'host': 'localhost',
            'user': 'root',
            'password': 'asd515359',
            'database': 'config_db',
            'port': 3306
        }
    
    try:
        config = configparser.ConfigParser()
        config.read(config_path, encoding='utf-8')
        
        if 'mysqld' not in config:
            print("警告: 配置文件中没有[mysqld]节，使用默认配置")
            return {
                'host': 'localhost',
                'user': 'root',
                'password': 'asd515359',
                'database': 'config_db',
                'port': 3306
            }
        
        db_config = {
            'host': config.get('mysqld', 'host', fallback='localhost'),
            'user': config.get('mysqld', 'user', fallback='root'),
            'password': config.get('mysqld', 'password', fallback='asd515359'),
            'database': config.get('mysqld', 'database', fallback='config_db'),
            'port': config.getint('mysqld', 'port', fallback=3306)
        }
        
        return db_config
        
    except Exception as e:
        print(f"警告: 读取配置文件失败 {e}，使用默认配置")
        return {
            'host': 'localhost',
            'user': 'root',
            'password': 'asd515359',
            'database': 'config_db',
            'port': 3306
        }


DB_CONFIG = load_db_config()

def getImgWidth(imagePath):
  img = Image.open(imagePath)
  img_w, _ = img.size
  return img_w
def getEntityName(cursor,ConfigID):
  idx = (ConfigID - 1) * 6 + 1
  cursor.execute("SELECT entityName FROM entities WHERE `index` = %s AND ConfigID = %s",(idx,ConfigID))
  result = cursor.fetchone()
  return result[0] if result else None
def matchLabels(imgdir,labeldir,dstdir,cursor):
  for cfg in imgdir.glob('*/'):
    cfg = int(cfg.name)
    cls = getEntityName(cursor,cfg)
    if not cls:
      print(f"ConfigID {cfg} 未找到对应的 cls")
      continue
    imageDir = imgdir / str(cfg)
    imageList = [imageFile for imageFile in imageDir.glob('*.bmp')]
    if not imageList:
      print(f"ConfigID {cfg} 没有找到对应的图片")
      continue
    img_w = getImgWidth(imageList[1])
    labelpath = labeldir / cls / str(img_w)
    dstlabel = dstdir / str(cfg)
    dstlabel.mkdir(parents=True,exist_ok=True)
    for txt in labelpath.glob('*.txt'):
      shutil.copy(txt,dstlabel)
    print(f"ConfigID {cfg} 的标签已复制到 {dstlabel}")

def main():
  conn = mysql.connector.connect(**DB_CONFIG)
  cursor = conn.cursor()
  rootdir = Path(__file__).resolve().parent
  imgdir = rootdir / 'images'
  labeldir = rootdir / 'template'
  dstdir = rootdir / 'labels'
  matchLabels(imgdir, labeldir, dstdir, cursor)
  cursor.close()
  conn.close()

if __name__ == '__main__':
  main()