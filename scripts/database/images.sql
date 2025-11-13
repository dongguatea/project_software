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

 Date: 17/08/2025 18:35:42
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for images
-- ----------------------------
DROP TABLE IF EXISTS `images`;
CREATE TABLE `images`  (
  `image_id` bigint(20) NOT NULL AUTO_INCREMENT,
  `theta` decimal(10, 4) NOT NULL,
  `phi` decimal(10, 4) NOT NULL,
  `R` int(11) NULL DEFAULT 2000,
  `ConfigID` bigint(20) NOT NULL,
  `image_path` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL,
  PRIMARY KEY (`image_id`) USING BTREE,
  INDEX `idx_angle`(`theta`, `phi`) USING BTREE,
  INDEX `idx_param`(`ConfigID`) USING BTREE,
  INDEX `idx_images_cfg_theta_phi`(`ConfigID`, `theta`, `phi`, `R`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 5898151 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
