CREATE TABLE IF NOT EXISTS `mc5_` (
  `m5id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `m4id` bigint unsigned NOT NULL,
  `aeid` bigint unsigned NOT NULL,
  `modl` varchar(5) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `hitc` double DEFAULT NULL,
  `coff` double DEFAULT NULL,
  PRIMARY KEY (`m5id`),
  UNIQUE KEY `m5id` (`m5id`),
  KEY `aeid` (`aeid`) USING BTREE,
  KEY `m4id` (`m4id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=7785612 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci