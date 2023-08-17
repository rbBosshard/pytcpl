import pandas as pd
import mysql.connector
import os

from src.utils.constants import INPUT_DIR_PATH, AEIDS_LIST_PATH
from src.utils.query_db import get_db_config


BUILD = 0
instances_total = 20

user, password, host, port, database = get_db_config()
connection = mysql.connector.connect(host=host, user=user, password=password, database=database)

destination_path = os.path.join(INPUT_DIR_PATH, f"candidate_aeids.parquet.gzip")


if BUILD:
    table = ["mc5"]
    query = f"SELECT aeid, " \
            f"COUNT(*) as count, " \
            f"SUM(hitc = 1) AS hitc_1_count, " \
            f"SUM(hitc = 0) AS hitc_0_count, " \
            f"SUM(hitc = 1) / COUNT(*) AS ratio " \
            f"FROM invitrodb_v3o5.mc5 " \
            f"GROUP BY aeid;"

    df_counts = pd.read_sql(query, connection)

    query = f"SELECT aeid, " \
            f"analysis_direction " \
            f"FROM invitrodb_v3o5.assay_component_endpoint " \
            f"WHERE analysis_direction='positive' " \
            f"AND signal_direction='gain';"

    df_analysis_direction = pd.read_sql(query, connection)
    df = df_counts.merge(df_analysis_direction, on="aeid", how="inner")
    df.to_parquet(destination_path, compression='gzip')
else:
    df = pd.read_parquet(destination_path)

df = df[df['count'] > 2000]
df = df[df['ratio'] > 0.005]
df = df.sort_values('hitc_1_count', ascending=False)

# Select the 'aeid' column from the DataFrame
aeids = df['aeid']

num_aeids = len(aeids)

# Calculate the number of tasks per worker
tasks_per_instance = (len(aeids) + instances_total - 1) // instances_total


def distribute_tasks(tasks, n):
    distributed_tasks = [[] for _ in range(n)]
    for i, task_id in enumerate(tasks):
        worker_idx = i % n
        distributed_tasks[worker_idx].append(task_id)
    return distributed_tasks


distributed_tasks = distribute_tasks(aeids, instances_total)

with open(AEIDS_LIST_PATH, "w") as file:
    for i, instance_tasks in enumerate(distributed_tasks):
        for task_id in instance_tasks[:tasks_per_instance]:
            file.write(str(task_id) + "\n")

print(f"Instances total: {instances_total}")
print(f"Total num aeids to process: {num_aeids}")
print(f"Num aeids per instance to process: {tasks_per_instance}")




