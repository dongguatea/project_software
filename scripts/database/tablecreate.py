import mysql.connector

DB_Config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'asd515359',
    'database': 'config_db',
    'port' : 3306
}

def main():
    #建立数据表
    sql = """
    CREATE TABLE IF NOT EXISTS images(
    image_id      BIGINT PRIMARY KEY AUTO_INCREMENT,
    theta   DECIMAL(10,4) NOT NULL,        
    phi   DECIMAL(10,4) NOT NULL,        
    R  INT DEFAULT 2000,
    ConfigID  BIGINT NOT NULL,               
    INDEX idx_angle (ConfigID,R,theta,phi),
    )ENGINE=InnoDB;
    CREATE TABLE IF NOT EXISTS eval_metrics (
    image_id     BIGINT NOT NULL,
    conf_thr     DECIMAL(5,3) NOT NULL,                 
    nms_iou_thr  DECIMAL(5,3) NOT NULL,                  
    match_iou    DECIMAL(5,3) NOT NULL,
    model       INT DEFAULT 0,                  
    tp           INT NOT NULL,
    fp           INT NOT NULL,
    fn           INT NOT NULL,
    n_pred       INT NOT NULL,
    n_gt         INT NOT NULL,
    PRIMARY KEY (image_id, conf_thr, nms_iou_thr, match_iou),
    CONSTRAINT fk_eval_img FOREIGN KEY (image_id) REFERENCES images(image_id)
    )ENGINE=InnoDB;
    CREATE TABLE IF NOT EXISTS param_sets (
    ConfigID BIGINT PRIMARY KEY,
    name varchar(255) NOT NULL,
    maxTemperature INT DEFAULT 70,
    minTemperature INT DEFAULT 0,
    hFOVPixels INT DEFAULT 640,
    vFOVPixels INT DEFAULT 512,
    hFOVDeg INT DEFAULT 20,
    vFOVDeg INT DEFAULT 16,
    percentBlur FLOAT DEFAULT 0,
    percentNoise FLOAT DEFAULT 0,
    entityName VARCHAR(255) NOT NULL,
    entityInter VARCHAR(255) NOT NULL,
    time VARCHAR(32) DEFAULT '12:00',
    hazeModel INT DEFAULT 3,
    rainRate INT DEFAULT 0
    )ENGINE=InnoDB;
    """
    conn = mysql.connector.connect(
        **DB_Config
    )
    cursor = conn.cursor()
    for stmt in sql.strip().split(';'):
        if stmt.strip():  # strip是用来去除字符串两端的空白部分
            cursor.execute(stmt)

    cursor.close()
    conn.close()

if __name__ == '__main__':
    main()