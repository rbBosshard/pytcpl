CREATE TABLE IF NOT EXISTS `output` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `aeid` int unsigned NOT NULL,
  `spid` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ac50` double DEFAULT NULL,
  `acc` double DEFAULT NULL,
  `actop` double DEFAULT NULL,
  `best_aic_model` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `conc` JSON,
  `fit_params` JSON,
  `hitcall` double DEFAULT NULL,
  `resp` JSON,
  `top` double DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `aeid` (`aeid`) USING BTREE,
  KEY `spid` (`spid`)
)
