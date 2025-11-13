/*
 Navicat Premium Data Transfer

 Source Server         : config_create
 Source Server Type    : MySQL
 Source Server Version : 50744
 Source Host           : localhost:3306
 Source Schema         : config_db

 Target Server Type    : MySQL
 Target Server Version : 50744
 File Encoding         : 65001

 Date: 18/08/2025 11:02:42
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for param_sets
-- ----------------------------
DROP TABLE IF EXISTS `param_sets`;
CREATE TABLE `param_sets`  (
  `ConfigID` bigint(20) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `maxTemperature` int(11) NULL DEFAULT 70,
  `minTemperature` int(11) NULL DEFAULT 0,
  `hFOVPixels` int(11) NULL DEFAULT 640,
  `vFOVPixels` int(11) NULL DEFAULT 512,
  `hFOVDeg` int(11) NULL DEFAULT 20,
  `vFOVDeg` int(11) NULL DEFAULT 16,
  `percentBlur` double NULL DEFAULT 0,
  `percentNoise` double NULL DEFAULT 0,
  `time` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '12:00',
  `hazeModel` int(11) NULL DEFAULT 3,
  `ismanbo` tinyint(1) NULL DEFAULT 0,
  `ispanmao` tinyint(1) NULL DEFAULT 0,
  `ishajimi` tinyint(1) NULL DEFAULT 0,
  `ispanbaobao` tinyint(1) NULL DEFAULT 0,
  `visibility` double NULL DEFAULT 0,
  `rainRate` INT NULL DEFAULT 0,
  PRIMARY KEY (`ConfigID`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 3241 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
