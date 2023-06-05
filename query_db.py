import pandas as pd
import yaml
import time
import mysql.connector as mysql
from sqlalchemy import create_engine, text


def get_db_config():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config_db = config['DATABASE']
        return config_db['USERNAME'], config_db['PASSWORD'], config_db['HOST'], config_db['PORT'], config_db['DB']


def get_sqlalchemy_engine():
    username, password, host, port, db = get_db_config()
    url = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{db}"
    try:
        engine = create_engine(url)
        return engine

    except Exception as error:
        print(f"Error connecting to MySQL: {error}")
        return None


def get_db_conn():
    username, password, host, port, db = get_db_config()

    try:
        db_conn = mysql.connect(
            host=host,
            user=username,
            password=password,
            port=port,
            database=db
        )
        return db_conn

    except Exception as error:
        print(f"Error connecting to MySQL: error: {error}")
        return None


def tcpl_query(query):
    try:
        if query.lower().startswith("delete"):
            db_conn = get_db_conn()
            mycursor = db_conn.cursor()
            mycursor.execute(query)
            db_conn.commit()
        else:
            engine = get_sqlalchemy_engine()
            start_time = time.time()
            df = pd.read_sql(text(query), con=engine.connect())
            print(f"Query {query[:100]} >> {df.shape[0]} rows >> {str(time.time() - start_time)} seconds.")
            return df
    except Exception as e:
        print(f"Error querying MySQL: {e}")
        return None
