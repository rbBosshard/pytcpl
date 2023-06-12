import os
import time

import mysql.connector as mysql
import pandas as pd
import yaml
import os
from sqlalchemy import create_engine, text


def get_db_config():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(ROOT_DIR, 'config.yaml')
    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        login_name = os.getlogin()
        config_db = config[login_name]['DATABASE']
        return config_db['USERNAME'], config_db['PASSWORD'], config_db['HOST'], config_db['PORT'], config_db['DB']


def get_sqlalchemy_engine():
    username, password, host, port, db = get_db_config()
    url = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{db}"
    try:
        return create_engine(url)
    except Exception as error:
        print(f"Error connecting to MySQL: {error}")
        return None


def get_db_conn():
    username, password, host, port, db = get_db_config()
    try:
        return mysql.connect(
            host=host,
            user=username,
            password=password,
            port=port,
            database=db)
    except mysql.Error as error:
        raise ConnectionError("Error connecting to MySQL: {}".format(error))


def tcpl_query(query, verbose=False):
    try:
        if query.lower().startswith("delete"):
            db_conn = get_db_conn()
            cursor = db_conn.cursor()
            cursor.execute(query)
            db_conn.commit()
        else:
            engine = get_sqlalchemy_engine()
            start_time = time.time()
            df = pd.read_sql(text(query), con=engine.connect())
            if verbose:
                print(f"Query {query[:100]} >> {df.shape[0]} rows >> {str(time.time() - start_time)} seconds.")
            return df
    except Exception as e:
        print(f"Error querying MySQL: {e}")
        return None
