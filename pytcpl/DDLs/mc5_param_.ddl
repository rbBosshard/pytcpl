CREATE TABLE IF NOT EXISTS `mc5_param_` (
  `m5id` bigint unsigned NOT NULL,
  `aeid` int NOT NULL,
  `hit_param` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `hit_val` double DEFAULT NULL,
  KEY `m5id` (`m5id`) USING BTREE,
  KEY `aeid` (`aeid`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci