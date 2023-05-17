import yaml
import mysql.connector

def get_db_conn():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config_db = config['DATABASE']

    try:
        db_conn =  mysql.connector.connect(
            host=config_db['HOST'],
            user=config_db['USERNAME'],
            password=config_db['PASSWORD'],
            database=config_db['DB']
        )
        return db_conn

    except Exception as error:
        print(f"Error connecting to MySQL: error: {error}")
        return None
    
   