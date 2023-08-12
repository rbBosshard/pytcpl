CREATE TABLE IF NOT EXISTS `output` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `aeid` int unsigned NOT NULL,
  `spid` varchar(25) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `dsstox_substance_id` varchar(15) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `chid` int unsigned NOT NULL,
  `best_aic_model` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `hitcall` double DEFAULT NULL,
  `ac50` double DEFAULT NULL,
  `acc` double DEFAULT NULL,
  `actop` double DEFAULT NULL,
  `top` double DEFAULT NULL,
  `conc` JSON,
  `resp` JSON,
  `fit_params` JSON,
  PRIMARY KEY (`id`),
  KEY `aeid` (`aeid`) USING BTREE,
  KEY `spid` (`spid`)
)
