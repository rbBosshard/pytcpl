CREATE TABLE IF NOT EXISTS `cutoff` (
  `aeid` int unsigned NOT NULL,
  `bmad` double NOT NULL,
  `bmed` double NOT NULL,
  `onesd` double NOT NULL,
  `cutoff` double NOT NULL,
  PRIMARY KEY (`aeid`),
  KEY `aeid` (`aeid`) USING BTREE
)

