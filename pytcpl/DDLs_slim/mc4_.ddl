CREATE TABLE IF NOT EXISTS `mc4_` (
  `m4id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `aeid` bigint unsigned NOT NULL,
  `spid` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `bmad` double NOT NULL,
  PRIMARY KEY (`m4id`),
  KEY `aeid` (`aeid`) USING BTREE,
  KEY `idx_mc4_spid` (`spid`)
) ENGINE=InnoDB AUTO_INCREMENT=6186299 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci

