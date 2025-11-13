#!/usr/bin/env python3
"""bulk_generate_multi_csv.py

按模板批量生成数据到 **新的 CSV 文件**，
文件名与原模板相同，存放在指定 --output-dir 目录。
ConfigID 起点取指定数据库表最大值 + 1。
"""

import csv, random, argparse
from pathlib import Path
import numpy as np
import mysql.connector
from itertools import product

# sensors
NAMES = ['Mid-wave Infarared - MWIR', 'Long Wave Infarared - LWIR']
MIN_MAX_TEMPERATURE_OPTIONS = [(20, 60), (0, 70), (0, 90)]
FOVPIXEL_OPTIONS = [(320, 256), (640, 512), (1024, 1024)]
# FOVDEG_OPTIONS = [(10, 8), (20, 16), (32, 32)]
ENTITYLIST = ['water','manbo']
SENSIM_BLUR_CHOICES = np.linspace(0, 1, 3).tolist()
SENSIM_NOISE_CHOICES = np.linspace(0, 1, 3).tolist()
numlist = []
trackFilePath = r'E:\track.trk'
# entities
ENTITYINTER  = [hajimi for hajimi in range(4)]
EntityListCoord = [(0,0,0),(0,0,0)]
EntityInterCoord = [(40,-40,0),(40,40,0),(-40,-40,0),(-40,40,0)]
GEOM_FILE_MAP = {'water' : 'object/water/water.flt',
                 'manbo':'object/manbo/manbo.flt',
                 'panmao':'object/panmao/panmao.flt',
                 'hajimi1':'object/hajimi/hajimiuntitled.flt',
                 'hajimi2':'object/hajimi/hajimiuntitled.flt',
                 'hajimi3':'object/hajimi/hajimiuntitled.flt',
                 'hajimi4':'object/hajimi/hajimiuntitled.flt',
                 }
MAT_FILE_MAP  = {
                 'water' : 'object/water/water.ms',
                 'manbo':'object/manbo/manboemat.ms', 
                 'panmao':'object/panmao/panmaoemat.ms',
                 'hajimi1':'object/hajimi/hajimiuntitled.ms',
                 'hajimi2':'object/hajimi/hajimiuntitled.ms',
                 'hajimi3':'object/hajimi/hajimiuntitled.ms',
                 'hajimi4':'object/hajimi/hajimiuntitled.ms',
                 }
TIME_CHOICES = ['0:00','12:00'] #['6:30','12:00','18:00','22:30']
HAZEMODEL = [3,9,0]
RAINRATE = [0.25] #小雨,中雨,大雨
VISIBILITY = [1500,750,400]
DATE_CHOICES = ['07/24/2024'] #[f'{m:02d}/15/2024' for m in range(1,13)]


def fetch_max_config_id(host,port,user,password,db,table):
    conn = mysql.connector.connect(host=host,port=port,user=user,
                                   password=password,database=db)
    cur = conn.cursor()
    cur.execute(f"SELECT MAX(ConfigID) FROM {table}")
    res = cur.fetchone()
    cur.close()
    conn.close()
    return int(res[0]) if res and res[0] is not None else 0

def load_template(p:Path):
    with p.open(newline='',encoding='utf-8-sig') as f:
        rdr=csv.DictReader(f)
        return rdr.fieldnames,list(rdr)

def ensure_dir(p:Path): p.mkdir(parents=True,exist_ok=True)

def main():
    filepath = Path(__file__).resolve()
    csvpath = filepath.parent / 'csv'
    outpath = filepath.parent / 'generate'
    ap=argparse.ArgumentParser()
    ap.add_argument('--sensors',default= csvpath / 'sensors.csv',type=Path)
    ap.add_argument('--entities',default=csvpath /'entities.csv',type=Path)
    ap.add_argument('--terrain', default=csvpath /'terrain.csv',type=Path)
    ap.add_argument('--track',   default=csvpath /'track.csv',type=Path)
    ap.add_argument('--scenario',default=csvpath /'scenario.csv',type=Path)
    ap.add_argument('--env',     default=csvpath /'environment-manager.csv',type=Path)
    # ap.add_argument('-n','--num',type=int,default=500)
    ap.add_argument('--output-dir',type=Path,default=outpath)
    ap.add_argument('--id-table',default='scenario')
    ap.add_argument('--db-host',default='localhost')
    ap.add_argument('--db-port',default=3306,type=int)
    ap.add_argument('--db-user',default='root')
    ap.add_argument('--db-password',default='asd515359')
    ap.add_argument('--db-name',default='config_db')
    args=ap.parse_args()


    start_id=fetch_max_config_id(args.db_host,args.db_port,args.db_user,
                                 args.db_password,args.db_name,args.id_table)+1

    print('起始 ConfigID =',start_id)

    sen_fields,sen_rows = load_template(args.sensors)
    e_fields,e_rows=load_template(args.entities)
    t_fields,t_rows=load_template(args.terrain)
    tr_fields,tr_rows=load_template(args.track)
    s_fields,s_rows=load_template(args.scenario)
    em_fields,em_rows=load_template(args.env)

    ent_tpl=e_rows[0]
    
    sensors_tpl=sen_rows[0];terrain_tpl=t_rows[0]; track_tpl=tr_rows[0]; scenario_tpl=s_rows[0]; env_tpl=em_rows[0]

    ensure_dir(args.output_dir)
    files={
        'sensors' : (args.output_dir/args.sensors.name,sen_fields),
        'entities': (args.output_dir/args.entities.name,e_fields),
        'terrain':  (args.output_dir/args.terrain.name,t_fields),
        'track':    (args.output_dir/args.track.name,tr_fields),
        'scenario': (args.output_dir/args.scenario.name,s_fields),
        'env':      (args.output_dir/args.env.name,em_fields),
    }
    writers,handles={},{}
    try:
        for k,(p,fields) in files.items():
            fh=p.open('w',newline='',encoding='utf-8')
            handles[k]=fh
            w=csv.DictWriter(fh,fieldnames=fields,extrasaction='ignore')
            w.writeheader(); writers[k]=w

        combos = list(product(
            NAMES, MIN_MAX_TEMPERATURE_OPTIONS, FOVPIXEL_OPTIONS,
            SENSIM_BLUR_CHOICES, SENSIM_NOISE_CHOICES,
            DATE_CHOICES,TIME_CHOICES,HAZEMODEL
        ))
        count = 0
# 特征参数组合 + 日期、时间、气溶胶
        for idx, (name, (mn, mx), (hp, vp), noise, blur,date,time,hazemodel) in enumerate(combos):
            index_counter = 0
            if hazemodel != 0:
                rainRate = 0
                cfg = start_id + count
                # entities.csv
                for entityname,(x,y,z) in zip(ENTITYLIST,EntityListCoord):
                    row = ent_tpl.copy()
                    if entityname in GEOM_FILE_MAP and entityname in MAT_FILE_MAP:
                        geomFileName = GEOM_FILE_MAP[entityname]
                        matSysFileName = MAT_FILE_MAP[entityname]
                    if entityname == 'water':
                        row.update({
                            'index' : str(index_counter),
                            'entityName'  : entityname,
                            'category' : 'water',
                            'worldXCoord' : x,
                            'worldYCoord' : y,
                            'worldZCoord' : z,
                            'ConfigID' : cfg
                        })
                    else:
                        row.update({
                            'index' : str(index_counter),
                            'entityName'  : entityname,
                            'category' : 'default category',
                            'worldXCoord' : x,
                            'worldYCoord' : y,
                            'worldZCoord' : z,
                            'ConfigID' : cfg,
                            'geomFileName': geomFileName,
                            'matSysFileName': matSysFileName
                        })
                    writers['entities'].writerow(row)
                    index_counter += 1

                for entityname, (x, y, z) in zip(ENTITYINTER, EntityInterCoord):
                    row = ent_tpl.copy()
                    if entityname in GEOM_FILE_MAP and entityname in MAT_FILE_MAP:
                        geomFileName = GEOM_FILE_MAP[entityname]
                        matSysFileName = MAT_FILE_MAP[entityname]
                    row.update({
                        'index'    : str(index_counter),
                        'entityName'     : entityname,
                        'category': 'default category',
                        'worldXCoord' : x, 'worldYCoord' : y, 'worldZCoord' : z,
                        'ConfigID' : cfg,
                        'geomFileName' : geomFileName,
                        'matSysFileName' : matSysFileName
                    })
                    writers['entities'].writerow(row)
                    index_counter += 1
                sensors = sensors_tpl.copy(); sensors['ConfigID'] = cfg
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
                    'trackFilePath' : trackFilePath
                })
                writers['sensors'].writerow(sensors)

                terr=terrain_tpl.copy(); terr['ConfigID']=cfg
                writers['terrain'].writerow(terr)

                tr=track_tpl.copy(); tr['ConfigID']=cfg
                tr['startTime'] = 0
                tr['endTime'] = 1439
                tr['saveFileName']=f'E:/bupt_IR/images/{cfg}/0.bmp'
                writers['track'].writerow(tr)

                sc=scenario_tpl.copy(); sc['ConfigID']=cfg
                writers['scenario'].writerow(sc)

                em=env_tpl.copy(); em['ConfigID']=cfg
                em['time']=time #random.choice(TIME_CHOICES)
                em['date']=date #random.choice(DATE_CHOICES)
                em['hazeModel']=hazemodel
                em['rainRate'] = rainRate
                em['visibility']=0
                writers['env'].writerow(em)
                count += 1
            elif hazemodel == 0:

                for rainRate in RAINRATE:
                    index_counter = 0
                    cfg = start_id + count
                    for entityname, (x, y, z) in zip(ENTITYLIST, EntityListCoord):
                        row = ent_tpl.copy()
                        if entityname in GEOM_FILE_MAP and entityname in MAT_FILE_MAP:
                            geomFileName = GEOM_FILE_MAP[entityname]
                            matSysFileName = MAT_FILE_MAP[entityname]
                        if entityname == 'water':
                            row.update({
                                'index': str(index_counter),
                                'entityName': entityname,
                                'category': 'water',
                                'worldXCoord': x,
                                'worldYCoord': y,
                                'worldZCoord': z,
                                'ConfigID': cfg
                            })
                        else:
                            row.update({
                                'index': str(index_counter),
                                'entityName': entityname,
                                'category': 'default category',
                                'worldXCoord': x,
                                'worldYCoord': y,
                                'worldZCoord': z,
                                'ConfigID': cfg,
                                'geomFileName': geomFileName,
                                'matSysFileName': matSysFileName
                            })
                        writers['entities'].writerow(row)
                        index_counter += 1

                    for entityname, (x, y, z) in zip(ENTITYINTER, EntityInterCoord):
                        row = ent_tpl.copy()
                        if entityname in GEOM_FILE_MAP and entityname in MAT_FILE_MAP:
                            geomFileName = GEOM_FILE_MAP[entityname]
                            matSysFileName = MAT_FILE_MAP[entityname]
                        row.update({
                            'index': str(index_counter),
                            'entityName': entityname,
                            'category': 'default category',
                            'worldXCoord': x, 'worldYCoord': y, 'worldZCoord': z,
                            'ConfigID': cfg,
                            'geomFileName': geomFileName,
                            'matSysFileName': matSysFileName
                        })
                        writers['entities'].writerow(row)
                        index_counter += 1
                    sensors = sensors_tpl.copy();
                    sensors['ConfigID'] = cfg
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
                        'trackFilePath': trackFilePath
                    })
                    writers['sensors'].writerow(sensors)

                    terr = terrain_tpl.copy();
                    terr['ConfigID'] = cfg
                    writers['terrain'].writerow(terr)

                    tr = track_tpl.copy();
                    tr['ConfigID'] = cfg
                    tr['startTime'] = 0
                    tr['endTime'] = 1439
                    tr['saveFileName'] = f'E:/bupt_IR/images/{cfg}/0.bmp'
                    writers['track'].writerow(tr)

                    sc = scenario_tpl.copy();
                    sc['ConfigID'] = cfg
                    writers['scenario'].writerow(sc)

                    em = env_tpl.copy()
                    if rainRate == 0.25:
                        visibility = VISIBILITY[0]
                    elif rainRate == 2 :
                        visibility = VISIBILITY[1]
                    elif rainRate == 8 :
                        visibility = VISIBILITY[2]
                    em['ConfigID'] = cfg
                    em['time'] = time  # random.choice(TIME_CHOICES)
                    em['date'] = date  # random.choice(DATE_CHOICES)
                    em['hazeModel'] = hazemodel
                    em['rainRate'] = rainRate
                    em['visibility'] = visibility
                    writers['env'].writerow(em)
                    count+=1
    finally:
        for fh in handles.values(): fh.close()

if __name__=='__main__':
    main()
