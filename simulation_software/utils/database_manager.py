"""
数据库管理模块
封装数据库连接和操作
"""
import mysql.connector
from mysql.connector import Error
import logging

class DatabaseManager:
    """数据库管理类"""
    
    def __init__(self, config=None):
        """
        初始化数据库管理器
        :param config: 数据库配置字典
        """
        self.config = config or {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': 'asd515359',
            'database': 'config_db',
            'charset': 'utf8',
            'autocommit': True
        }
        self.connection = None
        
    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            return True
        except Error as e:
            logging.error(f"数据库连接失败: {e}")
            return False
            
    def disconnect(self):
        """断开数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            
    def test_connection(self):
        """测试数据库连接"""
        try:
            conn = mysql.connector.connect(**self.config)
            if conn.is_connected():
                conn.close()
                return True
            return False
        except Error as e:
            logging.error(f"数据库连接测试失败: {e}")
            return False
            
    def execute_query(self, query, params=None):
        """执行查询"""
        try:
            if not self.connection or not self.connection.is_connected():
                if not self.connect():
                    return None
                    
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)
            result = cursor.fetchall()
            cursor.close()
            return result
        except Error as e:
            logging.error(f"查询执行失败: {e}")
            return None
            
    def execute_insert(self, query, data):
        """执行插入操作"""
        try:
            if not self.connection or not self.connection.is_connected():
                if not self.connect():
                    return False
                    
            cursor = self.connection.cursor()
            cursor.executemany(query, data)
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            logging.error(f"插入操作失败: {e}")
            return False
            
    def get_max_config_id(self, table='scenario'):
        """获取最大ConfigID"""
        try:
            query = f"SELECT MAX(ConfigID) as max_id FROM {table}"
            result = self.execute_query(query)
            if result and result[0]['max_id'] is not None:
                return int(result[0]['max_id'])
            return 0
        except Exception as e:
            logging.error(f"获取最大ConfigID失败: {e}")
            return 0
            
    def get_unused_config_ids(self):
        """获取未使用的ConfigID列表"""
        try:
            query = """
                SELECT ConfigID FROM scenario
                WHERE Use_ConfigID = 0
                ORDER BY ConfigID
            """
            result = self.execute_query(query)
            return [row['ConfigID'] for row in result] if result else []
        except Exception as e:
            logging.error(f"获取未使用ConfigID失败: {e}")
            return []
            
    def mark_config_used(self, config_id):
        """标记ConfigID为已使用"""
        try:
            if not self.connection or not self.connection.is_connected():
                if not self.connect():
                    return False
                    
            cursor = self.connection.cursor()
            query = "UPDATE scenario SET Use_ConfigID=1 WHERE ConfigID=%s"
            cursor.execute(query, (config_id,))
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            logging.error(f"标记ConfigID已使用失败: {e}")
            return False
            
    def get_table_columns(self, table_name):
        """获取表的列名"""
        try:
            query = f"SHOW COLUMNS FROM `{table_name}`"
            result = self.execute_query(query)
            return [row['Field'].lower() for row in result] if result else []
        except Exception as e:
            logging.error(f"获取表列名失败: {e}")
            return []
            
    def clear_temp_data(self, config_id):
        """清理临时数据"""
        try:
            if not self.connection or not self.connection.is_connected():
                if not self.connect():
                    return False
                    
            cursor = self.connection.cursor()
            
            # 删除相关数据
            tables = ['entities', 'sensors', 'terrain', 'track', 'scenario', 'environment-manager']
            for table in tables:
                try:
                    query = f"DELETE FROM `{table}` WHERE ConfigID=%s"
                    cursor.execute(query, (config_id,))
                except Error as e:
                    logging.warning(f"清理表{table}失败: {e}")
                    
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            logging.error(f"清理临时数据失败: {e}")
            return False
            
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()