CREATE TABLE IF NOT EXISTS `output` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `aeid` int unsigned NOT NULL,
  `spid` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `fit_model` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `cutoff` double DEFAULT NULL,
  `hitcall` double DEFAULT NULL,
  `ac50` double DEFAULT NULL,
  `bmd` double DEFAULT NULL,
  `top` double DEFAULT NULL,
  `concentration_unlogged` JSON,
  `response` JSON,
  `fitparams` JSON,
  PRIMARY KEY (`id`),
  KEY `aeid` (`aeid`) USING BTREE,
  KEY `spid` (`spid`)
)
