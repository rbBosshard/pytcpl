
  SELECT m.aeid, COUNT(*) AS count, normalized_data_type, burst_assay
  FROM invitrodb_v3o5.mc4 as m
  INNER JOIN assay_component_endpoint AS e ON m.aeid = e.aeid
  WHERE e.analysis_direction = 'positive'
  GROUP BY m.aeid
  HAVING COUNT(*) < 1000 and normalized_data_type = 'percent_activity' and burst_assay = 0

