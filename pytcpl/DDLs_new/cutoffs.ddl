CREATE TABLE IF NOT EXISTS `cutoffs` (
  `aeid` int unsigned NOT NULL,
  `bmad` double NOT NULL,
  `cutoff` double NOT NULL,
  PRIMARY KEY (`aeid`),
  KEY `aeid` (`aeid`) USING BTREE
)

