
-- 配置文件数据库，用于批量生成配置文件，并且用于后期的标注文件生成。

CREATE DATABASE IF NOT EXISTS config_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE config_db;

-- ========== Table: terrain ==========
CREATE TABLE IF NOT EXISTS terrain (
  ConfigID      VARCHAR(64) NOT NULL,
  name          VARCHAR(255) DEFAULT 'water',
  terrainFilepath     VARCHAR(255) NOT NULL,
  mtlsysFilepath      VARCHAR(255) NOT NULL,
  swOriginLatitude      FLOAT DEFAULT 21.5,
  swOriginLongitude     FLOAT DEFAULT -157.9,
  xPositionOffset     FLOAT DEFAULT 0,
  yPositionOffset     FLOAT DEFAULT 0,
  altitude        FLOAT DEFAULT 15,
  -- 可在此处添加特征列
  PRIMARY KEY (ConfigID)
) ENGINE=InnoDB;

-- ========== Table:environment_manager ==========
CREATE TABLE IF NOT EXISTS environment_manager (
  ConfigID      VARCHAR(64) NOT NULL,
  time        VARCHAR(32)  DEFAULT 6,
  date        VARCHAR(32)  DEFAULT '07/24/2004',
  celestialSkyTableFilepath VARCHAR(255)  DEFAULT './atm/stellar_sky.ir',
  atmosphereFile    VARCHAR(255)  DEFAULT './atm/summer.mcd',
  minAirTemperature INT DEFAULT 10,
  maxAirTemperature INT DEFAULT 20,
  atmModel  INT DEFAULT 1,
  hazeModel  INT DEFAULT 1,
  cloudModel INT DEFAULT 0,
  visibility  FLOAT DEFAULT 0,
-- 可在此处添加特征列
  PRIMARY KEY (ConfigID)
) ENGINE=InnoDB;

-- ========== Table: scenario ==========
CREATE TABLE IF NOT EXISTS scenario (
  ConfigID      VARCHAR(64) NOT NULL,
  useLitWindows BOOLEAN     DEFAULT 0,
  channelClipNear INT DEFAULT 1,
  channelClipFar INT DEFAULT 120000,
  PRIMARY KEY (ConfigID)
) ENGINE=InnoDB;

-- ========== Table: track ==========
CREATE TABLE IF NOT EXISTS track (
  ConfigID      VARCHAR(64) NOT NULL,
  startTime         FLOAT DEFAULT 0,
  endTime        FLOAT DEFAULT 30,
  saveFileName    VARCHAR(256) NOT NULL,
  -- frameRate, startTime, endTime, loop, save, FileName ...
  PRIMARY KEY (ConfigID)
) ENGINE=InnoDB;

-- ========== Table: entities (variable amount per config) ==========
CREATE TABLE IF NOT EXISTS entities (
  ConfigID      VARCHAR(64) NOT NULL,
  entity_index    INT NOT NULL,
  entityName     VARCHAR(255) NOT NULL,
  geomFileName    VARCHAR(255) NOT NULL,
  matSysFileName   VARCHAR(255) NOT NULL,
  category      VARCHAR(255) DEFAULT 'defaultCategory',
  worldXCoord     FLOAT DEFAULT 0,
  worldYCoord     FLOAT DEFAULT 0,
  worldZCoord     FLOAT DEFAULT 0,
  latitude      FLOAT DEFAULT 21,
  longitude     FLOAT  DEFAULT -157,
  altitude     FLOAT DEFAULT 0,
  heading      FLOAT DEFAULT -90,
  pitch       FLOAT DEFAULT 0,
  roll       FLOAT DEFAULT 0,
  PRIMARY KEY (ConfigID, entity_index)
) ENGINE=InnoDB;

-- ========== Table: sensors_defaulats(variable amount per config) ==========
CREATE TABLE IF NOT EXISTS sensor_defaults (
  name   VARCHAR(64)  PRIMARY KEY,   -- 例如 'A' 或 'B'
  ax INT NOT NULL,
  bx INT NOT NULL,
  nx INT NOT NULL,
  senSim_optics_haloIntensity INT NOT NULL,
  senSim_eletronics_maxAGCgain INT NOT NULL
) ENGINE=InnoDB;

INSERT INTO sensor_defaults(name,ax,bx,nx,senSim_optics_haloIntensity,senSim_eletronics_maxAGCgain)
VALUES
  ('Mid-wave Infarared - MWIR',3,5,1,24,100),
  ('Long Wave Infarared - LWIR',8,12,1,30,1)
ON DUPLICATE KEY UPDATE
  ax = VALUES(ax),
  bx = VALUES(bx),
  nx = VALUES(nx),
  senSim_optics_haloIntensity = VALUES(senSim_optics_haloIntensity),
  senSim_eletronics_maxAGCgain = VALUES(senSim_eletronics_maxAGCgain);

-- ========== Table: sensors (variable amount per config) ==========
CREATE TABLE IF NOT EXISTS sensors (
  ConfigID     VARCHAR(64) NOT NULL,
  name   VARCHAR(64) NOT NULL,
  sensor_index     INT DEFAULT 0,
  ax INT DEFAULT NULL,
  bx INT DEFAULT NULL,
  nx INT DEFAULT NULL,
  maxLightLevel FLOAT DEFAULT 0.003,
  maxTemperature FLOAT NOT NULL,
  minTemperature FLOAT NOT NULL,
  worldXCoord     FLOAT DEFAULT 2451,
  worldYCoord     FLOAT DEFAULT 1504,
  worldZCoord     FLOAT DEFAULT 11.413,
  latitude FLOAT DEFAULT 21.513,
  longitude FLOAT DEFAULT -157.876235990755,
  altitude FLOAT DEFAULT 11.41,
  hFOVPixels FLOAT NOT NULL,
  vFOVPixels FLOAT NOT NULL,
  vFOVDeg FLOAT NOT NULL,
  hFOVDeg FLOAT NOT NULL,
  isActive INT DEFAULT 1,
  senSim_percentBlur FLOAT DEFAULT 1,
  senSim_percentNoise FLOAT DEFAULT 1,
  PRIMARY KEY (ConfigID)
) ENGINE=InnoDB;
ALTER TABLE sensors ADD COLUMN senSim_eletronics_maxAGCgain VARCHAR(64) DEFAULT NULL,
          ADD COLUMN senSim_optics_haloIntensity VARCHAR(64) DEFAULT NULL;
CREATE INDEX idx_sensors_name ON sensors(name);
DELIMITER //
CREATE TRIGGER sensors_bi
BEFORE INSERT ON sensors
FOR EACH ROW
BEGIN
  DECLARE d_ax,d_bx,d_nx,d_senSim_optics_haloIntensity,d_senSim_eletronics_maxAGCgain INT;
  SELECT ax,bx,nx,senSim_optics_haloIntensity,senSim_eletronics_maxAGCgain
  INTO d_ax,d_bx,d_nx,d_senSim_optics_haloIntensity,d_senSim_eletronics_maxAGCgain FROM sensor_defaults
  WHERE name = new.name;
  SET NEW.ax        = IFNULL(NEW.ax,d_ax);
  SET NEW.bx        = IFNULL(NEW.bx,d_bx);
  SET NEW.nx      = IFNULL(NEW.nx,d_nx);
  SET NEW.senSim_eletronics_maxAGCgain = IFNULL(NEW.senSim_eletronics_maxAGCgain,d_senSim_eletronics_maxAGCgain);
  SET NEW.senSim_optics_haloIntensity = IFNULL(NEW.senSim_optics_haloIntensity,d_senSim_optics_haloIntensity);
END//
DELIMITER;
ALTER TABLE terrain
  MODIFY COLUMN name VARCHAR(255) DEFAULT 'water',
  MODIFY COLUMN xPositionOffset FLOAT DEFAULT 0,
  MODIFY COLUMN yPositionOffset FLOAT DEFAULT 0;
ALTER TABLE scenario
  MODIFY COLUMN useLitWindows BOOLEAN DEFAULT 0;
-- INSERT INTO track(ConfigID,startTime,endTime,saveFileName) VALUES(
--   1,0,229,'E:\filetest'
-- );
-- INSERT INTO terrain(ConfigID,name,terrainFilepath,mtlsysFilepath) VALUES(
--   1,'water','E:\IR_image\1.txt','E:\IR_image\eee.txt'
-- );
-- INSERT INTO sensors(ConfigID,sensor_name,minTemperature,maxTemperature,hFOVPixels,vFOVPixels,hFOVDeg,vFOVDeg) VALUES(
--   1,'Long Wave Infarared - LWIR',0,60,1024,1024,32,32
-- )
-- INSERT INTO sensors(ConfigID,sensor_name,minTemperature,maxTemperature,hFOVPixels,vFOVPixels,hFOVDeg,vFOVDeg) VALUES(
--   3,'Long Wave Infarared - LWIR',0,60,1024,1024,32,32
-- )
-- INSERT INTO scenario(ConfigID) VALUES(
--   1
-- );
-- INSERT INTO environment_manager(ConfigID,time,date,hazeModel,cloudModel,visibility) VALUES(
--   1,'6:45','10/13/2021',1,0,1
-- );
-- INSERT INTO entities(ConfigID,entity_index,entityName,geomFileName,matSysFileName) VALUES(
--   1,0,'DDG','E:\IR_image\1.txt','E:\IR_image\eee.txt'
-- )

