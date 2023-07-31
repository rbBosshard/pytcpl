CREATE TABLE IF NOT EXISTS `mc4_agg_` (
  `aeid` bigint unsigned NOT NULL,
  `m0id` bigint unsigned NOT NULL,
  `m1id` bigint unsigned NOT NULL,
  `m2id` bigint unsigned NOT NULL,
  `m3id` bigint unsigned NOT NULL,
  `m4id` bigint unsigned NOT NULL,
  KEY `aeid` (`aeid`),
  KEY `m0id` (`m0id`),
  KEY `m1id` (`m1id`),
  KEY `m2id` (`m2id`),
  KEY `m3id` (`m3id`),
  KEY `m4id` (`m4id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci