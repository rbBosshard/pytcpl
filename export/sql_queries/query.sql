DROP PROCEDURE IF EXISTS create_grouped_data_table;

DELIMITER //
CREATE PROCEDURE create_grouped_data_table()
BEGIN
	-- This command will unset the SQL mode and allow the query to execute without strict GROUP BY restrictions.
	SET sql_mode = ''; 
    SET @digits = 3;
	-- Drop the temporary table if it already exists
	DROP TEMPORARY TABLE IF EXISTS grouped_data;
	-- Create the temporary table
	CREATE TEMPORARY TABLE grouped_data AS (
		SELECT 
        c.chid, 
        d.dsstox_substance_id,
        d.chnm, COUNT(*) AS group_count,
        ROUND(AVG(b.hitc), @digits) AS hitcall_avg,
        ROUND(MAX(b.hitc) - MIN(b.hitc), @digits) AS hitcall_range,
        ROUND(STD(b.hitc), @digits) AS hitcall_std,
        GROUP_CONCAT(b.hitc) AS hitcall_list
		FROM mc4_ AS a
		INNER JOIN mc5_ AS b ON a.m4id = b.m4id
		INNER JOIN sample AS c ON a.spid = c.spid
		INNER JOIN chemical AS d ON c.chid = d.chid
		WHERE a.aeid = 784
		GROUP BY chid);
	-- SELECT the table
    SELECT * FROM grouped_data ORDER BY group_count DESC;
END //
DELIMITER ;

-- Call the stored procedure to drop and create the temporary table
CALL create_grouped_data_table();

SELECT
  group_count,
  COUNT(*) AS group_count_frequency
FROM grouped_data
GROUP BY group_count
ORDER BY group_count_frequency DESC;


SELECT avg(group_count)
FROM grouped_data;


