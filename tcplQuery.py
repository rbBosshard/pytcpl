import mysql.connector
import pandas as pd

def tcplQuery(query):
    # connect to mysql
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="invitrodb_v3o5")
    except:
        # logging.error("Error connecting to MySQL")
        print("Error connecting to MySQL")
        return None
    
    # query select from table

    df = pd.read_sql_query(query, db)
    db.close() #close the connection
    return df