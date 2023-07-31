import os

import mysql.connector as mysql
import pandas as pd
import yaml
from sqlalchemy import create_engine, text


def get_db_config():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(ROOT_DIR, 'config/db_login.yaml')
    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        login_name = os.getlogin()
        config_db = config["db_login"][login_name]
        return config_db['username'], config_db['password'], config_db['host'], config_db['port'], config_db['db']


def get_sqlalchemy_engine():
    username, password, host, port, db = get_db_config()
    url = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{db}"
    try:
        return create_engine(url)
    except Exception as error:
        print(f"Error connecting to MySQL: {error}")
        return None


def query_db(query):
    try:
        if any(query.lower().startswith(x) for x in ["delete", "create", "drop"]):
            user, pw, host, port, db = get_db_config()
            db_conn = mysql.connect(host=host, user=user, password=pw, port=port, database=db)
            cursor = db_conn.cursor()
            cursor.execute(query)
            db_conn.commit()
        else:
            engine = get_sqlalchemy_engine()
            df = pd.read_sql(text(query), con=engine.connect())
            return df
    except Exception as e:
        print(f"Error querying MySQL: {e}")
        return None
