import os
import urllib.parse
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# Load environment variables
load_dotenv()

def get_engine():
    # Allow overriding with a direct connection string
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        db_user = os.getenv("DB_USER", "churn_user")
        db_password = os.getenv("DB_PASSWORD", "StrongPassword123")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "3306")
        db_name = os.getenv("DB_NAME", "churn_intelligence")
        
        encoded_password = urllib.parse.quote_plus(db_password)
        db_url = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        
    return create_engine(db_url)

def save_df_to_sql(df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
    engine = get_engine()
    df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)

def read_sql_to_df(query_or_table: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(query_or_table, con=engine)
