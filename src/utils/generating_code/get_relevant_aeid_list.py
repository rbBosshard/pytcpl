import os
import mysql.connector
import pandas as pd
from src.utils.constants import INPUT_DIR_PATH, AEIDS_LIST_PATH
from src.utils.query_db import get_db_config


def distribute_aeids_to_instances(tasks, total_instances):
    """
    Distributes AEIDs (Assay Endpoint IDs) to instances for parallel processing.

    Args:
        tasks (list): List of AEIDs to be distributed.
        total_instances (int): Total number of instances for parallel processing.

    Returns:
        list of lists: Distributed AEID tasks for each instance.
    """
    distributed_tasks = [[] for _ in range(total_instances)]
    for i, task_id in enumerate(tasks):
        worker_idx = i % total_instances
        distributed_tasks[worker_idx].append(task_id)
    return distributed_tasks


def main():
    user, password, host, port, database = get_db_config()
    connection = mysql.connector.connect(host=host, user=user, password=password, database=database)

    destination_path = os.path.join(INPUT_DIR_PATH, f"candidate_aeids.parquet.gzip")
    instances_total = 4

    BUILD = 0

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

    aeids = df['aeid']
    num_aeids = len(aeids)
    tasks_per_instance = (len(aeids) + instances_total - 1) // instances_total

    distributed_tasks = distribute_aeids_to_instances(aeids, instances_total)

    with open(AEIDS_LIST_PATH, "w") as file:
        for i, instance_tasks in enumerate(distributed_tasks):
            for task_id in instance_tasks[:tasks_per_instance]:
                file.write(str(task_id) + "\n")

    print(f"Instances total: {instances_total}")
    print(f"Total num aeids to process: {num_aeids}")
    print(f"Num aeids per instance to process: {tasks_per_instance}")


if __name__ == "__main__":
    main()
