"""
应用程序配置文件
"""
import os
from pathlib import Path

# 应用程序基本信息
APP_NAME = "仿真软件管理系统"
APP_VERSION = "1.0.0"
ORGANIZATION_NAME = "BUPT Simulation Lab"

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
XMLCREATE_DIR = SCRIPTS_DIR / "xmlcreate"

# 数据库默认配置
DEFAULT_DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'asd515359',
    'database': 'config_db',
    'charset': 'utf8',
    'autocommit': True
}

# 日志配置
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'simulation.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# 仿真参数默认值
DEFAULT_SIMULATION_PARAMS = {
    'sensor': {
        'names': ['Mid-wave Infrared - MWIR', 'Long Wave Infrared - LWIR'],
        'temperature_ranges': [(20, 60), (0, 70), (0, 90)],
        'fov_pixel_options': [(320, 256), (640, 512), (1024, 1024)],
        'noise_range': (0.0, 1.0),
        'blur_range': (0.0, 1.0)
    },
    'entity': {
        'entity_list': ['water', 'manbo'],
        'interference_entities': ['hajimi1', 'hajimi2', 'hajimi3', 'hajimi4'],
        'default_coords': [(0, 0, 0), (0, 0, 0)],
        'interference_coords': [(40, -40, 0), (40, 40, 0), (-40, -40, 0), (-40, 40, 0)]
    },
    'environment': {
        'time_choices': ['0:00', '12:00'],
        'date_choices': ['07/24/2024'],
        'haze_models': [0, 3, 9],
        'rain_rates': [0.25],
        'visibility_options': [1500, 750, 400]
    }
}

# 文件路径映射
GEOM_FILE_MAP = {
    'water': 'object/water/water.flt',
    'manbo': 'object/manbo/manbo.flt',
    'panmao': 'object/panmao/panmao.flt',
    'hajimi1': 'object/hajimi/hajimiuntitled.flt',
    'hajimi2': 'object/hajimi/hajimiuntitled.flt',
    'hajimi3': 'object/hajimi/hajimiuntitled.flt',
    'hajimi4': 'object/hajimi/hajimiuntitled.flt',
}

MAT_FILE_MAP = {
    'water': 'object/water/water.ms',
    'manbo': 'object/manbo/manboemat.ms',
    'panmao': 'object/panmao/panmaoemat.ms',
    'hajimi1': 'object/hajimi/hajimiuntitled.ms',
    'hajimi2': 'object/hajimi/hajimiuntitled.ms',
    'hajimi3': 'object/hajimi/hajimiuntitled.ms',
    'hajimi4': 'object/hajimi/hajimiuntitled.ms',
}