CREATE TABLE IF NOT EXISTS `mc4_param_` (
  `m4id` bigint unsigned NOT NULL,
  `aeid` int NOT NULL,
  `model` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `model_param` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `model_val` double DEFAULT NULL,
  KEY `m4id` (`m4id`) USING BTREE,
  KEY `aeid` (`aeid`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci