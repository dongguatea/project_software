#!/usr/bin/env python
# db2xml.py
# --------------------------------------------------
# 从数据库读取 → 替换模板节点 → 生成 test_001.xml …
# --------------------------------------------------
import os, copy,sys
import math
import mysql.connector
from lxml import etree
from copy import deepcopy

DB_CONFIG = dict(
    host="localhost",
    port=3306,
    user="root",
    password="asd515359",
    database="config_db",
    charset ="utf8",
    autocommit=True           # 自动提交，方便脚本反复执行
)

TEMPLATE = "template.xml"
OUT_DIR  = ".\out_xml"

# 一张表一行：全部节点名称 = 列名
SINGLE_ROW_TABLES = ["terrain", "environment-manager", "track", "scenario"]

# 列表型表：表名 → 模板中的父节点 XPath
ENTITY_TABLE = "entities"
SENSOR_TABLE = "sensors"

# ---------- 建库连接 ----------
try:
    conn = mysql.connector.connect(**DB_CONFIG)
except mysql.connector.Error as e:
    sys.exit(f"无法连接数据库: {e}")
cur = conn.cursor(dictionary=True)

# ---------- 取Use_ConfigID为0的ConfigID ----------
cur.execute("""
            SELECT ConfigID FROM scenario
            WHERE Use_ConfigID = 0
            ORDER BY ConfigID
            """ )
cfg_ids = [row["ConfigID"] for row in cur.fetchall()]
# ---------- 准备模板 ----------
tmpl_tree = etree.parse(TEMPLATE)
os.makedirs(OUT_DIR, exist_ok=True)

for idx, cfg in enumerate(cfg_ids, start=1):
    tree = copy.deepcopy(tmpl_tree)
    root = tree.getroot()

    # ---- 单行表 ----
    for tbl in SINGLE_ROW_TABLES:
        cur.execute(f"SELECT * FROM `{tbl}` WHERE ConfigID=%s LIMIT 1;", (cfg,))
        row = cur.fetchone()
        if not row:
            continue
        for col, val in row.items():
            if col in("ConfigID","Use_ConfigID") or val is None:
                continue
            node = root.find(f".//{tbl}/{col}")
            if node is not None:
                node.text = str(val)
                cur.execute("UPDATE scenario SET Use_ConfigID=1 WHERE ConfigID=%s;",(cfg,))

    # ---- entities ----
    cur.execute("SELECT * FROM entities WHERE ConfigID=%s ORDER BY `index`;", (cfg,))
    entities = cur.fetchall()
    tpl_elist = root.find('.//entity-list')
    tpl_entity = tpl_elist.find('entity') if tpl_elist is not None else None
    tpl_children = []
    if tpl_entity is not None:
        for ch in tpl_entity:
            tpl_children.append(deepcopy(ch))
    elist = tpl_elist or etree.SubElement(root, 'entity-list')
    elist.clear()
    for r in entities:  # entities = cur.fetchall()
        # 3.1 拷贝一个“空骨架”<entity>
        new_ent = etree.SubElement(elist, 'entity')
        # 3.2 先恢复所有模板子节点
        for tpl_ch in tpl_children:
            new_ent.append(deepcopy(tpl_ch))
        # 3.3 再用数据库字段覆盖或追加
        for col, val in r.items():
            if val is None or col in ('ConfigID', 'entity_index'):
                continue
            node = new_ent.find(col)
            if node is None:
                node = etree.SubElement(new_ent, col)
            node.text = str(val)

    # ---- sensors + sensim 合并 ----
    cur.execute("""
           SELECT * FROM sensors
           WHERE ConfigID=%s
           ORDER BY `index`
       """, (cfg,))
    sensors = cur.fetchall()
    tpl_sensor = root.find('.//sensor-list/sensor')
    tpl_sensim = None
    if tpl_sensor is not None:
        for child in tpl_sensor:
            if child.tag.lower() == 'sensim':
                tpl_sensim = child
                break

    # 2. 把 sensim 下面的所有默认子节点深拷贝到一个列表
    tpl_sensim_children = []
    if tpl_sensim is not None:
        for c in tpl_sensim:
            tpl_sensim_children.append(deepcopy(c))

    # 3. 清空旧的 sensor-list
    slist = root.find('.//sensor-list')
    if slist is None:
        slist = etree.SubElement(root, 'sensor-list')
    slist.clear()

    # 4. 针对每一行数据库 sensors，复制模板并再写入字段
    for r in sensors:  # sensors = cur.fetchall()
        # 4.1 复制一个完整的模板 <sensor>
        new_sensor = deepcopy(tpl_sensor)
        # 如果模板没写 type，就手动补上
        # if 'sensor_type' in r and r['sensor_type'] is not None:
        #     new_sensor.attrib['type'] = str(r['sensor_type'])

        # 4.2 拿到它自己的 sensim，清空里面所有默认子节点
        new_sensim = None
        for child in new_sensor:
            if child.tag == 'senSim':
                new_sensim = child
                break
        if new_sensim is None:
            new_sensim = etree.SubElement(new_sensor, 'senSim')

        # 4.3 先清空，再还原模板里的所有子节点
        new_sensim.clear()
        for tpl_child in tpl_sensim_children:
            new_sensim.append(deepcopy(tpl_child))

        # 4.4 再根据数据库列覆盖或追加
        for col, val in r.items():
            if val is None or col in ('ConfigID', 'sensor_index', 'sensor_type'):
                continue
            col_l = col.lower()

            if col_l.startswith('sensim_optics_'):
                tag = col[len('senSim_optics_'):]
                # 确保 optics 节点存在
                optics = new_sensim.find('optics')
                if optics is None:
                    optics = etree.SubElement(new_sensim, 'optics')
                etree.SubElement(optics, tag).text = str(val)

            if col_l.startswith('sensim_eletronics_'):
                tag = col[len('senSim_eletronics_'):]
                elec = new_sensim.find('eletronics')
                if elec is None:
                    elec = etree.SubElement(new_sensim, 'eletronics')
                etree.SubElement(elec, tag).text = str(val)

            if col_l.startswith('sensim_'):
                tag = col[len('senSim_'):]
                node = new_sensim.find(tag)
                if node is None:
                    node = etree.SubElement(new_sensim, tag)
                node.text = str(val)

            else:
                # 其余字段写到 <sensor> 下
                node = new_sensor.find(col)
                if node is None:
                    node = etree.SubElement(new_sensor, col)
                node.text = str(val)

        # 4.5 把新 sensor 加回列表
        slist.append(new_sensor)
    slist.tail = '\n'  # 换行美化
    # ---- 写文件 ----
    out_path = os.path.join(OUT_DIR, f"test_{cfg:03d}.xml")
    etree.indent(tree, space="    ", level=0)
    tree.write(out_path, encoding="utf-8", pretty_print=True, xml_declaration=False)
    print(f"✔ 生成 {out_path}")

cur.close()
conn.close()
print("全部 XML 生成完毕")
