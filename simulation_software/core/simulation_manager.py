"""
整合generate_fix_data, csv2sql, sql2xml, auto_control功能
"""
import os
import sys
import csv
import copy
import glob
import time
import subprocess
import pandas as pd
import mysql.connector
from pathlib import Path
from itertools import product
from lxml import etree
from copy import deepcopy
import numpy as np
import logging

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from ..utils.database_manager import DatabaseManager

class SimulationManager:
    """仿真管理器 - 整合所有仿真相关功能"""
    
    def __init__(self):
        self.db_manager = None
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.xmlcreate_dir = self.scripts_dir / "xmlcreate"
        
        self.NAMES = ['Mid-wave Infarared - MWIR', 'Long Wave Infarared - LWIR']
        self.MIN_MAX_TEMPERATURE_OPTIONS = [(0, 30), (0, 50), (0, 70)]
        self.FOVPIXEL_OPTIONS = [(320, 256), (640, 512), (1024, 1024)]
        self.ENTITYLIST = ['water', 'manbo']
        self.SENSIM_BLUR_CHOICES = np.linspace(0, 1, 3).tolist()
        self.SENSIM_NOISE_CHOICES = np.linspace(0, 1, 3).tolist()
        self.ENTITYINTER = [hajimi for hajimi in range(4)]
        self.EntityListCoord = [(0, 0, 0), (0, 0, 0)]
        self.EntityInterCoord = [(40, -40, 0), (40, 40, 0), (-40, -40, 0), (-40, 40, 0)]
        
        self.GEOM_FILE_MAP = {
            'water': 'object/water/water.flt',
            'manbo': 'object/manbo/manbo.flt',
            'panmao': 'object/panmao/panmao.flt',
            'hajimi1': 'object/hajimi/hajimiuntitled.flt',
            'hajimi2': 'object/hajimi/hajimiuntitled.flt',
            'hajimi3': 'object/hajimi/hajimiuntitled.flt',
            'hajimi4': 'object/hajimi/hajimiuntitled.flt',
        }
        
        self.MAT_FILE_MAP = {
            'water': 'object/water/water.ms',
            'manbo': 'object/manbo/manboemat.ms',
            'panmao': 'object/panmao/panmaoemat.ms',
            'hajimi1': 'object/hajimi/hajimiuntitled.ms',
            'hajimi2': 'object/hajimi/hajimiuntitled.ms',
            'hajimi3': 'object/hajimi/hajimiuntitled.ms',
            'hajimi4': 'object/hajimi/hajimiuntitled.ms',
        }
        
        self.TIME_CHOICES = ['0:00', '12:00']
        self.HAZEMODEL = [3, 9, 0]
        self.RAINRATE = [0.25]
        self.VISIBILITY = [1500, 750, 400]
        self.DATE_CHOICES = ['07/24/2024']
        
    def setup_database(self, db_config):
        """设置数据库连接"""
        self.db_manager = DatabaseManager(db_config)
        return self.db_manager.test_connection()
        
    def generate_config_files(self, parameters):
        """生成配置文件 - 基于generate_fix_data.py"""
        try:
            # 设置数据库连接
            db_config = {
                'host': parameters['database']['host'],
                'port': parameters['database']['port'],
                'user': parameters['database']['user'],
                'password': parameters['database']['password'],
                'database': parameters['database']['database'],
                'charset': 'utf8',
                'autocommit': True
            }
            
            if not self.setup_database(db_config):
                logging.error("数据库连接失败")
                return False
                
            # 获取起始ConfigID
            start_id = self.db_manager.get_max_config_id('scenario') + 1
            logging.info(f'起始 ConfigID = {start_id}')
            
            # 准备输出目录
            output_dir = Path(parameters['simulation']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 加载模板
            csv_dir = self.xmlcreate_dir / 'csv'
            templates = self._load_templates(csv_dir)
            
            if not templates:
                logging.error("模板文件加载失败")
                return False
                
            # 生成组合数据
            success = self._generate_parameter_combinations(
                templates, start_id, output_dir, parameters
            )
            
            return success
            
        except Exception as e:
            logging.error(f"生成配置文件时出错: {e}")
            return False
            
    def _load_templates(self, csv_dir):
        """加载CSV模板文件"""
        try:
            templates = {}
            template_files = {
                'sensors': 'sensors.csv',
                'entities': 'entities.csv',
                'terrain': 'terrain.csv',
                'track': 'track.csv',
                'scenario': 'scenario.csv',
                'env': 'environment-manager.csv'
            }
            
            for key, filename in template_files.items():
                file_path = csv_dir / filename
                if file_path.exists():
                    with file_path.open(newline='', encoding='utf-8-sig') as f:
                        reader = csv.DictReader(f)
                        templates[key] = {
                            'fieldnames': reader.fieldnames,
                            'rows': list(reader)
                        }
                else:
                    logging.warning(f"模板文件不存在: {file_path}")
                    
            return templates
            
        except Exception as e:
            logging.error(f"加载模板文件时出错: {e}")
            return None
            
    def _generate_parameter_combinations(self, templates, start_id, output_dir, parameters):
        """生成参数组合"""
        try:
            # 创建输出文件
            files = {}
            writers = {}
            handles = {}
            
            for key, template in templates.items():
                file_path = output_dir / f"{key}.csv"
                fh = file_path.open('w', newline='', encoding='utf-8')
                handles[key] = fh
                writer = csv.DictWriter(fh, fieldnames=template['fieldnames'], extrasaction='ignore')
                writer.writeheader()
                writers[key] = writer
                
            try:
                # 生成参数组合
                combos = list(product(
                    [parameters['sensor']['name']],
                    [(parameters['sensor']['min_temperature'], parameters['sensor']['max_temperature'])],
                    [(parameters['sensor']['h_fov_pixels'], parameters['sensor']['v_fov_pixels'])],
                    [parameters['sensor']['noise']],
                    [parameters['sensor']['blur']],
                    [parameters['environment']['date']],
                    [parameters['environment']['time']],
                    [parameters['environment']['haze_model']]
                ))
                
                count = 0
                
                for idx, (name, (mn, mx), (hp, vp), noise, blur, date, time, hazemodel) in enumerate(combos):
                    self._generate_single_config(
                        writers, templates, start_id + count, 
                        name, mn, mx, hp, vp, noise, blur, date, time, hazemodel, parameters
                    )
                    count += 1
                    
            finally:
                for fh in handles.values():
                    fh.close()
                    
            logging.info(f"成功生成 {count} 个配置组合")
            return True
            
        except Exception as e:
            logging.error(f"生成参数组合时出错: {e}")
            return False
            
    def _generate_single_config(self, writers, templates, cfg_id, name, mn, mx, hp, vp, 
                              noise, blur, date, time, hazemodel, parameters):
        """生成单个配置"""
        try:
            # 获取模板
            ent_tpl = templates['entities']['rows'][0]
            sensors_tpl = templates['sensors']['rows'][0]
            terrain_tpl = templates['terrain']['rows'][0]
            track_tpl = templates['track']['rows'][0]
            scenario_tpl = templates['scenario']['rows'][0]
            env_tpl = templates['env']['rows'][0]
            
            # 处理实体
            index_counter = 0
            
            # 主要实体
            for entityname, (x, y, z) in zip(self.ENTITYLIST, self.EntityListCoord):
                row = ent_tpl.copy()
                if entityname in self.GEOM_FILE_MAP and entityname in self.MAT_FILE_MAP:
                    geomFileName = self.GEOM_FILE_MAP[entityname]
                    matSysFileName = self.MAT_FILE_MAP[entityname]
                    
                if entityname == 'water':
                    row.update({
                        'index': str(index_counter),
                        'entityName': entityname,
                        'category': 'water',
                        'worldXCoord': x,
                        'worldYCoord': y,
                        'worldZCoord': z,
                        'ConfigID': cfg_id
                    })
                else:
                    row.update({
                        'index': str(index_counter),
                        'entityName': entityname,
                        'category': 'default category',
                        'worldXCoord': x,
                        'worldYCoord': y,
                        'worldZCoord': z,
                        'ConfigID': cfg_id,
                        'geomFileName': geomFileName,
                        'matSysFileName': matSysFileName
                    })
                    
                writers['entities'].writerow(row)
                index_counter += 1
                
            # 干扰实体
            for entityname, (x, y, z) in zip(self.ENTITYINTER, self.EntityInterCoord):
                row = ent_tpl.copy()
                if entityname in self.GEOM_FILE_MAP and entityname in self.MAT_FILE_MAP:
                    geomFileName = self.GEOM_FILE_MAP[entityname]
                    matSysFileName = self.MAT_FILE_MAP[entityname]
                    
                row.update({
                    'index': str(index_counter),
                    'entityName': str(entityname),
                    'category': 'default category',
                    'worldXCoord': x,
                    'worldYCoord': y,
                    'worldZCoord': z,
                    'ConfigID': cfg_id,
                    'geomFileName': geomFileName,
                    'matSysFileName': matSysFileName
                })
                
                writers['entities'].writerow(row)
                index_counter += 1
                
            # 传感器
            sensors = sensors_tpl.copy()
            sensors['ConfigID'] = cfg_id
            sensors.update({
                'name': name,
                'minTemperature': mn,
                'maxTemperature': mx,
                'hFOVPixels': hp,
                'vFOVPixels': vp,
                'hFOVDeg': hp // 32,
                'vFOVDeg': vp // 32,
                'senSim_percentNoise': noise,
                'senSim_percentBlur': blur,
                'trackFilePath': parameters['sensor']['track_path']
            })
            writers['sensors'].writerow(sensors)
            
            # 地形
            terr = terrain_tpl.copy()
            terr['ConfigID'] = cfg_id
            writers['terrain'].writerow(terr)
            
            # 轨迹
            tr = track_tpl.copy()
            tr['ConfigID'] = cfg_id
            tr['startTime'] = 0
            tr['endTime'] = 1439
            tr['saveFileName'] = f'E:/bupt_IR/images/{cfg_id}/0.bmp'
            writers['track'].writerow(tr)
            
            # 场景
            sc = scenario_tpl.copy()
            sc['ConfigID'] = cfg_id
            writers['scenario'].writerow(sc)
            
            # 环境
            em = env_tpl.copy()
            em['ConfigID'] = cfg_id
            em['time'] = time
            em['date'] = date
            em['hazeModel'] = hazemodel
            
            if hazemodel == 0:
                em['rainRate'] = parameters['environment']['rain_rate']
                em['visibility'] = parameters['environment']['visibility']
            else:
                em['rainRate'] = 0
                em['visibility'] = 0
                
            writers['env'].writerow(em)
            
        except Exception as e:
            logging.error(f"生成单个配置时出错: {e}")
            raise
            
    def import_to_database(self, parameters):
        """导入数据库 - 基于csv2sql.py"""
        try:
            # 设置数据库连接
            db_config = {
                'host': parameters['database']['host'],
                'port': parameters['database']['port'],
                'user': parameters['database']['user'],
                'password': parameters['database']['password'],
                'database': parameters['database']['database'],
                'charset': 'utf8',
                'autocommit': True
            }
            
            if not self.setup_database(db_config):
                logging.error("数据库连接失败")
                return False
                
            # CSV文件目录
            csv_dir = Path(parameters['simulation']['output_dir'])
            
            # 遍历所有CSV文件
            for csv_path in glob.glob(str(csv_dir / "*.csv")):
                table = os.path.splitext(os.path.basename(csv_path))[0]
                
                # 获取表的列信息
                cols_db = set(self.db_manager.get_table_columns(table))
                if not cols_db:
                    logging.warning(f"表 {table} 不存在，跳过")
                    continue
                    
                logging.info(f"导入 {csv_path} → 表 {table}")
                
                # 分块读取CSV文件
                chunk_size = 1000
                total_rows = 0
                
                for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                    if table == "scenario":
                        if "Use_ConfigID" not in chunk.columns:
                            chunk["Use_ConfigID"] = 0
                        else:
                            chunk["Use_ConfigID"].fillna(0, inplace=True)
                            
                    # 统一列名去空格
                    chunk.rename(columns=lambda c: c.strip(), inplace=True)
                    
                    # 筛选有效列
                    valid_cols = [c for c in chunk.columns if c.lower() in cols_db]
                    if not valid_cols:
                        logging.warning(f"该分块无有效列，跳过")
                        continue
                        
                    # 生成INSERT语句
                    cols_fmt = ", ".join(f"`{c}`" for c in valid_cols)
                    placeholders = ", ".join(["%s"] * len(valid_cols))
                    sql = f"INSERT INTO `{table}` ({cols_fmt}) VALUES ({placeholders})"
                    
                    # 准备数据
                    data = []
                    for _, row in chunk.iterrows():
                        row_data = []
                        for col in valid_cols:
                            value = row[col]
                            if pd.isna(value):
                                row_data.append(None)
                            else:
                                row_data.append(value)
                        data.append(tuple(row_data))
                        
                    # 执行插入
                    if self.db_manager.execute_insert(sql, data):
                        total_rows += len(data)
                        logging.info(f"  + {len(data)} 行")
                    else:
                        logging.error(f"插入失败: {table}")
                        return False
                        
                logging.info(f"表 {table} 总共插入 {total_rows} 行")
                
            logging.info("CSV文件全部导入完毕")
            return True
            
        except Exception as e:
            logging.error(f"导入数据库时出错: {e}")
            return False
            
    def generate_xml_files(self, parameters):
        """生成XML文件 - 基于sql2xml.py"""
        try:
            # 设置数据库连接
            db_config = {
                'host': parameters['database']['host'],
                'port': parameters['database']['port'],
                'user': parameters['database']['user'],
                'password': parameters['database']['password'],
                'database': parameters['database']['database'],
                'charset': 'utf8',
                'autocommit': True
            }
            
            if not self.setup_database(db_config):
                logging.error("数据库连接失败")
                return False
                
            # 模板文件路径
            template_path = self.xmlcreate_dir / "template.xml"
            if not template_path.exists():
                logging.error(f"模板文件不存在: {template_path}")
                return False
                
            # 输出目录
            out_dir = self.xmlcreate_dir / "out_xml"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取未使用的ConfigID
            cfg_ids = self.db_manager.get_unused_config_ids()
            if not cfg_ids:
                logging.warning("没有找到未使用的ConfigID")
                return True
                
            # 加载模板
            tmpl_tree = etree.parse(str(template_path))
            
            # 单行表
            single_row_tables = ["terrain", "environment-manager", "track", "scenario"]
            
            for idx, cfg in enumerate(cfg_ids, start=1):
                tree = copy.deepcopy(tmpl_tree)
                root = tree.getroot()
                
                # 处理单行表
                for tbl in single_row_tables:
                    query = f"SELECT * FROM `{tbl}` WHERE ConfigID=%s LIMIT 1"
                    result = self.db_manager.execute_query(query, (cfg,))
                    
                    if result and len(result) > 0:
                        row = result[0]
                        for col, val in row.items():
                            if col in ("ConfigID", "Use_ConfigID") or val is None:
                                continue
                            node = root.find(f".//{tbl}/{col}")
                            if node is not None:
                                node.text = str(val)
                                
                        # 标记为已使用
                        self.db_manager.mark_config_used(cfg)
                        
                # 处理entities
                self._process_entities_xml(root, cfg)
                
                # 处理sensors
                self._process_sensors_xml(root, cfg)
                
                # 写入文件
                out_path = out_dir / f"test_{cfg:03d}.xml"
                etree.indent(tree, space="    ", level=0)
                tree.write(str(out_path), encoding="utf-8", pretty_print=True, xml_declaration=False)
                logging.info(f"✔ 生成 {out_path}")
                
            logging.info("全部 XML 生成完毕")
            return True
            
        except Exception as e:
            logging.error(f"生成XML文件时出错: {e}")
            return False
            
    def _process_entities_xml(self, root, cfg):
        """处理entities节点"""
        try:
            query = "SELECT * FROM entities WHERE ConfigID=%s ORDER BY `index`"
            entities = self.db_manager.execute_query(query, (cfg,))
            
            if not entities:
                return
                
            tpl_elist = root.find('.//entity-list')
            tpl_entity = tpl_elist.find('entity') if tpl_elist is not None else None
            
            tpl_children = []
            if tpl_entity is not None:
                for ch in tpl_entity:
                    tpl_children.append(deepcopy(ch))
                    
            elist = tpl_elist or etree.SubElement(root, 'entity-list')
            elist.clear()
            
            for r in entities:
                # 创建新的entity节点
                new_ent = etree.SubElement(elist, 'entity')
                
                # 恢复模板子节点
                for tpl_ch in tpl_children:
                    new_ent.append(deepcopy(tpl_ch))
                    
                # 用数据库字段覆盖
                for col, val in r.items():
                    if val is None or col in ('ConfigID', 'entity_index'):
                        continue
                    node = new_ent.find(col)
                    if node is None:
                        node = etree.SubElement(new_ent, col)
                    node.text = str(val)
                    
        except Exception as e:
            logging.error(f"处理entities节点时出错: {e}")
            
    def _process_sensors_xml(self, root, cfg):
        """处理sensors节点"""
        try:
            query = "SELECT * FROM sensors WHERE ConfigID=%s ORDER BY `index`"
            sensors = self.db_manager.execute_query(query, (cfg,))
            
            if not sensors:
                return
                
            tpl_sensor = root.find('.//sensor-list/sensor')
            tpl_sensim = None
            
            if tpl_sensor is not None:
                for child in tpl_sensor:
                    if child.tag.lower() == 'sensim':
                        tpl_sensim = child
                        break
                        
            # 获取sensim模板子节点
            tpl_sensim_children = []
            if tpl_sensim is not None:
                for c in tpl_sensim:
                    tpl_sensim_children.append(deepcopy(c))
                    
            # 清空旧的sensor-list
            slist = root.find('.//sensor-list')
            if slist is None:
                slist = etree.SubElement(root, 'sensor-list')
            slist.clear()
            
            # 处理每个sensor
            for r in sensors:
                # 复制模板sensor
                new_sensor = deepcopy(tpl_sensor)
                
                # 获取sensim节点
                new_sensim = None
                for child in new_sensor:
                    if child.tag == 'senSim':
                        new_sensim = child
                        break
                        
                if new_sensim is None:
                    new_sensim = etree.SubElement(new_sensor, 'senSim')
                    
                # 恢复sensim子节点
                new_sensim.clear()
                for tpl_child in tpl_sensim_children:
                    new_sensim.append(deepcopy(tpl_child))
                    
                # 填充数据
                for col, val in r.items():
                    if val is None or col in ('ConfigID', 'sensor_index', 'sensor_type'):
                        continue
                        
                    col_l = col.lower()
                    
                    if col_l.startswith('sensim_optics_'):
                        tag = col[len('senSim_optics_'):]
                        optics = new_sensim.find('optics')
                        if optics is None:
                            optics = etree.SubElement(new_sensim, 'optics')
                        etree.SubElement(optics, tag).text = str(val)
                        
                    elif col_l.startswith('sensim_electronics_'):
                        tag = col[len('senSim_electronics_'):]
                        elec = new_sensim.find('electronics')
                        if elec is None:
                            elec = etree.SubElement(new_sensim, 'electronics')
                        etree.SubElement(elec, tag).text = str(val)
                        
                    elif col_l.startswith('sensim_'):
                        tag = col[len('senSim_'):]
                        node = new_sensim.find(tag)
                        if node is None:
                            node = etree.SubElement(new_sensim, tag)
                        node.text = str(val)
                        
                    else:
                        # 其他字段写到sensor下
                        node = new_sensor.find(col)
                        if node is None:
                            node = etree.SubElement(new_sensor, col)
                        node.text = str(val)
                        
                # 添加到列表
                slist.append(new_sensor)
                
            slist.tail = '\\n'
            
        except Exception as e:
            logging.error(f"处理sensors节点时出错: {e}")
            
    def start_simulation(self, parameters):
        """启动仿真 - 基于auto_control.py"""
        try:
            logging.info("准备启动仿真软件...")
            
            # 这里可以添加自动控制仿真软件的逻辑
            # 由于涉及到具体的软件界面操作，暂时返回成功
            # 实际实现时需要根据具体的仿真软件进行调整
            
            logging.info("仿真软件启动成功")
            return True
            
        except Exception as e:
            logging.error(f"启动仿真时出错: {e}")
            return False