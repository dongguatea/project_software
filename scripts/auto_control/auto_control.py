import csv
import time
import subprocess
import pyautogui
import cv2
import os
from pyautogui import ImageNotFoundException
# 等待时间统一设置
SHORT_DELAY = 0.5
LONG_DELAY = 2.0

current_dir = os.path.dirname(__file__)
dir_root = os.path.dirname(current_dir)
xml_create_dir = os.path.join(dir_root, 'xmlcreate')
# CSV 文件路径
CSV_PATH = os.path.join(xml_create_dir,'generated','Radar.csv')

# Sea值对应的按钮坐标，根据实际界面调整
Sea_POSITIONS = {
    0: (200, 700),
    1: (200, 720),
    2: (200, 740),
    3: (200, 760),
    4: (200, 780),
    5: (200, 800),
    6: (200, 820),
    7: (200, 840)
}

# 确保软件窗口获得焦点
def focus_window():
    # 如果需要用命令行启动软件，取消并修改下面路径
    # subprocess.Popen(r'C:\Path\to\YourApp.exe', shell=True)
    time.sleep(10)
    # 点击左上角固定位置以获得焦点
    pyautogui.click(158, 210)
    time.sleep(SHORT_DELAY)

# 检查复选框选中状态，需要安装 opencv: pip install opencv-python
def is_checked(template_path, region=None, confidence=0.9):
    try:
        loc = pyautogui.locateOnScreen(template_path, region=region, confidence=confidence)
        return loc is not None
    except ImageNotFoundException:
        # PyAutoGUI 抛出的“没找到图”异常
        return False

def input_row_vars(type,startazimuth,endazimuth,frequency,
                           startgrazing,endgrazing,rangeResolution,wideResolution,
                           Quantization,PRF,boolOblique,Sea):
    # 点击“路径选择”按钮 (213, 336)
    pyautogui.click(213, 336)
    time.sleep(LONG_DELAY)  # 等待文件管理系统窗口弹出

    # 在文件管理系统窗口中，点击输入框并输入路径,选择目标模型路径
    pyautogui.click(123,456 )
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.write(r'C:\path\to\your\folder')  # 替换为实际文件夹路径
    time.sleep(SHORT_DELAY)
    pyautogui.click(123,123)
    # 点击“选择文件夹”按钮 (816, 678) 关闭对话框
    pyautogui.click(1132, 881)
    time.sleep(LONG_DELAY)

    # 输入观测范围（方位角）
    values_group1 = [str(startazimuth),'20',str(endazimuth)]  # 请替换为实际要输入的数字
    positions_group1 = [(251, 404), (339, 404), (425, 404)]
    for pos, val in zip(positions_group1, values_group1):
        pyautogui.click(*pos)
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.write(val)
        time.sleep(SHORT_DELAY)

    #  输入观测范围（擦地角）
    values_group2 = [startgrazing, '5', endgrazing]
    positions_group2 = [(242, 434), (346, 434), (454, 434)]
    for pos, val in zip(positions_group2, values_group2):
        pyautogui.click(*pos)
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.write(val)
        time.sleep(SHORT_DELAY)

    # 根据 Sea 值点击不同按钮
    Sea_label = Sea
    if Sea_label in Sea_POSITIONS:
        pyautogui.click(*Sea_POSITIONS[Sea_label])
    else:
        # 如果未匹配到特定 san 值，可点击默认位置或跳过
        default_pos = (245, 701)  # san 对应默认位置
        pyautogui.click(*default_pos)
    time.sleep(SHORT_DELAY)


    # 选择识别类型（type）
    # pyautogui.click(244, 503)  # 点击复选框
    # time.sleep(SHORT_DELAY)
    # pyautogui.click(240, 531)  # 选择1
    # time.sleep(SHORT_DELAY)
    # pyautogui.click(241, 556)  # 选择2
    # time.sleep(SHORT_DELAY)

    # 输入频率、方位角、擦地角
    values_group3 = [frequency, startazimuth,'20',endazimuth, startgrazing, '5', endgrazing]
    positions_group3 = [
        (430, 637), (247, 670), (331, 669),
        (452, 668), (245, 701), (340, 698), (450, 701)
    ]
    for pos, val in zip(positions_group3, values_group3):
        pyautogui.click(*pos)
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.write(val)
        time.sleep(SHORT_DELAY)

    # 点击选项卡 (320, 779)
    pyautogui.click(320, 779)
    time.sleep(SHORT_DELAY)

    # 在新选项卡中四个位置输入数字（第四组）
    values_group4 = [rangeResolution,rangeResolution, wideResolution, wideResolution]
    positions_group4 = [(336, 819), (335, 850), (453, 822), (453, 852)]
    for pos, val in zip(positions_group4, values_group4):
        pyautogui.click(*pos)
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.write(val)
        time.sleep(SHORT_DELAY)

    #app7中可以调量化步长
    values_group5 = [Quantization,'10','30']
    positions_group5 = [(123,456),(111,123),(114,514)]
    for pos, val in zip(positions_group5, values_group5):
        pyautogui.click(*pos)
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.write(val)
        time.sleep(SHORT_DELAY)

    # 是否斜地投影
    checkbox_template = 'checkbox.png'
    checkbox_pos = (123,456)
    region = (checkbox_pos[0] - 10, checkbox_pos[1] - 10, 30, 30)
    desired = str(boolOblique).lower() in ('1', 'true', 'yes', '勾选', '选中')
    if desired:
        if not is_checked(checkbox_template, region=region):
            pyautogui.click(*checkbox_pos)
            time.sleep(SHORT_DELAY)
    else:
        if is_checked(checkbox_template, region=region):
            pyautogui.click(*checkbox_pos)
            time.sleep(SHORT_DELAY)

    #PRF 在代码中进行修改
    # 点击开始计算按钮 (394, 913)
    pyautogui.click(394, 913)

def main():
    focus_window()

    with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader, start=1):
            # 将 CSV 列内容赋给变量
            type          = row.get('type','')
            startazimuth  = row.get('startazimuth', '')
            endazimuth        = row.get('endazimuth', '')
            frequency     = row.get('frequency', '')
            startgrazing    = row.get('startgrazing', '')
            endgrazing      = row.get('endgrazing', '')
            rangeResolution  = row.get('rangeResolution', '')
            wideResolution       = row.get('wideResolution', '')
            Quantization       = row.get('Quantization', '')
            PRF        = row.get('PRF', '')
            boolOblique  = row.get('boolOblique', '')
            Sea        = row.get('Sea', '')
            Angular_reverse    = row.get('Angular_reverse', '')
            Foils       = row.get('Foils', '')

            print(f"Processing row {idx}: type={type}, sea={Sea}, fre={frequency}, ...")

            # 调用输入函数
            input_row_vars(type,startazimuth,endazimuth,frequency,
                           startgrazing,endgrazing,rangeResolution,wideResolution,
                           Quantization,PRF,boolOblique,Sea)

    print("所有操作已完成！")

if __name__ == '__main__':
    main()