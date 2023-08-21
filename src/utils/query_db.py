import os

import mysql.connector as mysql
import pandas as pd
import yaml
from sqlalchemy import create_engine, text

from src.utils.constants import CONFIG_DIR_PATH


def get_db_config():
    """
    Retrieve database configuration parameters from a YAML file.

    Reads the database configuration file based on the user's login name and extracts the necessary parameters.

    Returns:
        tuple: A tuple containing the username, password, host, port, and database name for the MySQL connection.
    """
    with open(os.path.join(CONFIG_DIR_PATH, 'config_db.yaml'), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        login_name = os.getlogin()
        config_db = config[login_name]
        return config_db['username'], config_db['password'], config_db['host'], config_db['port'], config_db['db']


def get_sqlalchemy_engine():
    """
    Create a SQLAlchemy database engine.

    Constructs and returns an SQLAlchemy engine object using the retrieved database configuration parameters.

    Returns:
        sqlalchemy.engine.base.Engine: The SQLAlchemy database engine object.
    """
    username, password, host, port, db = get_db_config()
    url = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{db}"
    try:
        return create_engine(url)
    except Exception as error:
        print(f"Error connecting to MySQL: {error}")
        return None


def query_db(query):
    """
    Execute a SQL query on a MySQL database.

    Executes the provided SQL query on a MySQL database. If the query is a DELETE, CREATE, or DROP statement,
    it is executed using the mysql.connector library. For other queries, an SQLAlchemy engine is used, and the
    results are returned as a pandas DataFrame.

    Args:
        query (str): The SQL query to be executed.

    Returns:
        pandas.DataFrame or None: If the query is a SELECT statement, returns the query results as a DataFrame.
                                 Otherwise, returns None.
    """
    try:
        if any(query.lower().startswith(x) for x in ["delete", "create", "drop"]):
            user, pw, host, port, db = get_db_config()
            db_conn = mysql.connect(host=host, user=user, password=pw, port=port, database=db)
            cursor = db_conn.cursor()
            cursor.execute(query)
            db_conn.commit()
            db_conn.close()
        else:
            engine = get_sqlalchemy_engine()
            con = engine.connect()
            df = pd.read_sql(text(query), con=con)
            con.close()
            engine.dispose()
            return df
    except Exception as e:
        print(f"Error querying MySQL: {e}")
        return None
