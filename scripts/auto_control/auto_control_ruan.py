"""
run_batch_from_xml.py
批量读取 out_xml/*.xml 并驱动 GUI 完成:
  1) 加载配置文件
  2) 加载对应轨迹文件
  3) 播放并保存图片
依赖: pip install pyautogui
"""

import glob, os, time, subprocess, sys
import pyautogui as pag
import win32gui

# ---------- 路径与常量 ----------
SOFTWARE_EXE = r"C:\Users\Public\Desktop\Steam.lnk"
OUT_XML_DIR = r"F:\data_manager\pythonProject1\out_xml"
TRACK_FILE = r"D:\Tracks\default_track.xml"
LOG_FILE     = r"processed.log"
SHORT_DELAY = 0.5
LONG_DELAY = 30.0


def is_lock():
  """
  检测 Windows 是否锁屏
  原理：查找锁屏界面窗口是否存在
  """
  hwnd = win32gui.FindWindow("Windows.UI.Core.CoreWindow","Lock")
  if hwnd != 0:
      return True
  else:
      return False
def monitor_and_run(automation_func, check_interval=0.2, mouse_threshold=5):
  """
  监控鼠标移动并在解锁后执行自动化操作
  :param automation_func: 要执行的自动化函数
  :param check_interval: 检查间隔（秒）
  :param mouse_threshold: 鼠标移动阈值（像素）
  """

  xpos,ypos = pag.position()
  try:
    while True:
      if is_lock():
        print("检测到锁屏，等待解锁...")
        sys.exit(0)
      x,y = pag.position()
      if abs(x - xpos) > mouse_threshold:
        print("检测到鼠标移动，停止自动化操作...")
        sys.exit(0)
      xpos = x
      ypos = y
      automation_func()
  except KeyboardInterrupt:
    print("用户中断，退出程序。")
    sys.exit(0)


# 软件内按钮/输入框坐标（示例像素）
BTN_NO = (319, 434)
BTN_LOAD_CONFIG = (283, 129)
BTN_LOAD_CONFIG_MANAGER = (313,576)  #加载配置文件的输入框坐标
BTN_RUN_SCENE = (301, 102)
TAB_SCENARIO = (106, 195)
BTN_LOAD_TRACK = (412, 512)
CHECK_ENABLED = (211, 729)
CHECK_SAVE_PIC = (206, 630)
BTN_PLAY = (92, 796)
FILE_DIALOG_OK = (890, 610)    # “打开”按钮

# 读取日志
processed = []
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        processed = [l.strip() for l in f if l.strip()]

# 扫描所有 XML
xml_files = sorted(glob.glob(os.path.join(OUT_XML_DIR, '*.xml')))

# 只保留“日志里没出现过”的文件
xml_to_process = [
    p for p in xml_files
    if os.path.basename(p) not in processed
]

if not xml_to_process:
    print("没有新的 XML，退出。")
    exit(0)

for xml_path in xml_to_process:
    name = os.path.basename(xml_path)
    print(f"处理 {name} ……")

    # ---------- 启动软件 ----------
    subprocess.Popen(SOFTWARE_EXE, shell=True)
    time.sleep(LONG_DELAY)  # 等界面完全加载

    # 点击左上角图标确保窗口获取焦点（示例）
    pag.click(70, 63)
    time.sleep(SHORT_DELAY)

    # ===== 导入配置文件 =====
    pag.click(*BTN_NO)                 # “NO” / Start Import
    time.sleep(SHORT_DELAY)

    pag.click(*BTN_LOAD_CONFIG)        # 打开文件对话框
    time.sleep(SHORT_DELAY)
    pag.click(*BTN_LOAD_CONFIG_MANAGER) #点击文件管理系统输入
    time.sleep(SHORT_DELAY)
    pag.hotkey('ctrl', 'a')
    time.sleep(SHORT_DELAY)
    pag.write(xml_path)
    time.sleep(SHORT_DELAY)
    pag.click(*FILE_DIALOG_OK)
    time.sleep(LONG_DELAY)             # 等配置加载完成
    # pag.click(326,479)
    # time.sleep(SHORT_DELAY)
    pag.click(246,480)
    time.sleep(LONG_DELAY)

    # ===== 导入轨迹 =====
    # pag.click(*BTN_RUN_SCENE)          # 进入场景界面
    # time.sleep(SHORT_DELAY)
    pag.click(*TAB_SCENARIO)
    time.sleep(SHORT_DELAY)
    pag.click(*BTN_LOAD_TRACK)         # 打开轨迹对话框
    time.sleep(SHORT_DELAY)
    pag.click(*BTN_LOAD_CONFIG_MANAGER) #点击文件管理系统输入
    time.sleep(SHORT_DELAY)
    pag.hotkey('ctrl', 'a')
    time.sleep(SHORT_DELAY)
    pag.write(TRACK_FILE)
    time.sleep(SHORT_DELAY)
    pag.click(*FILE_DIALOG_OK)
    time.sleep(SHORT_DELAY)

    # ===== 3. 播放 & 保存图片 =====
    # pag.click(*CHECK_ENABLED)          # 勾选 Enabled
    # time.sleep(SHORT_DELAY)
    # pag.click(*CHECK_SAVE_PIC)         # 勾选 Save Picture
    # time.sleep(SHORT_DELAY)
    pag.click(*BTN_PLAY)               # 点击播放
    time.sleep(30)             # 让场景跑一段时间
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(name + '\n')
    print(f"已记录：{name}")
    
print("所有 XML 处理完毕，日志已清空。")
# ===== 5.等级 =====
#0：225，567 1：225，613 2：225，635 3：225，663 4：225，685 5：225，710 6：214，732
