import yfinance as yf
import os
import pandas as pd
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pandas_datareader import data as pdr
import boto3
from pathlib import Path
from sqlalchemy import create_engine
from io import BytesIO
from airflow.operators.python_operator import PythonOperator
import psycopg2

FRED_API_KEY = os.getenv('FRED_API_KEY')
# set the start date and end date
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# set the logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for Airflow
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 17),
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
}
# Define DAG (Directed Acyclic Graph)
dag = DAG(
    'gold_etl',  # name of dag 
    default_args=default_args,
    description='Extract Gold & Bitcoin Data & US_Index',
    schedule ='@daily',  
    catchup=False,
)

def extract_data():
    data_path = os.path.join(os.getenv("AIRFLOW_HOME", "/opt/airflow"), "data")
    os.makedirs(data_path, exist_ok=True)
    lists = ['GC=F', 'BTC-USD', 'DX-Y.NYB']
    lists_file_name = ["gold_data", "bitcoin_data", "us_index_data"]
    
    for i in range(0, len(lists)):
        stock = yf.Ticker(lists[i])
        df = stock.history(start=START_DATE, end=END_DATE)

        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')

        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])

        df = df[['Date', 'Open', 'High', 'Low', 'Close']]

        df.dropna(inplace=True)
        print(df.head())
        df.to_csv(os.path.join(data_path, f"{lists_file_name[i]}.csv"), index=False)     
def extract_economic_indicators():
    logger.info("Extracting Economic Indicators from FRED...")
    indicators = {
        'CPIAUCSL': 'CPI',
        'FEDFUNDS': 'FedFunds',
        'UNRATE': 'Unemployment'
    }

    data_path = os.path.join(os.getenv("AIRFLOW_HOME", "/opt/airflow"), "data")
    os.makedirs(data_path, exist_ok=True)

    for code, name in indicators.items():
        try:
            df = pdr.DataReader(code, 'fred', START_DATE, END_DATE, api_key=FRED_API_KEY)
            df.to_csv(f"{data_path}/{name}_data.csv")
            logger.info(f"✅ Saved {name} data from FRED")
        except Exception as e:
            logger.error(f"❌ Failed to fetch {name}: {str(e)}")
# upload to minio(Data Lake)
def upload_to_minio():
    logger.info("Uploading CSV files to MinIO...")

    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv('S3_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )

    bucket = "gold-etl"
    data_path = os.path.join(os.getenv("AIRFLOW_HOME", "/opt/airflow"), "data")

    # check if bucket exists
    try:
        s3.head_bucket(Bucket=bucket)
    except:
        s3.create_bucket(Bucket=bucket)

    # whole csv file upload to MiniO
    for file in Path(data_path).glob("*.csv"):
        object_name = f"gold_data/{file.name}"
        with open(file, "rb") as f:
            s3.upload_fileobj(f, bucket, object_name)
            logger.info(f"✅ Uploaded {file.name} to MinIO bucket {bucket}/{object_name}")
def read_csv(bucket, key):
    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv('S3_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df
# insert data to postgres
def insert_gold_model_data():
    # Step 1: load the csv files from minIO
    # Yfinance: Gold, Bitcoin, Us_index 
    data_path = os.path.join(os.getenv("AIRFLOW_HOME", "/opt/airflow"), "data")
    gold = pd.read_csv(os.path.join(data_path, "gold_data.csv"))
    print(gold.columns)
    print(gold.head())

    bitcoin = pd.read_csv(os.path.join(data_path, "bitcoin_data.csv"))
    print(bitcoin.columns)
    print(bitcoin.head())
    
    us_index = pd.read_csv(os.path.join(data_path, "us_index_data.csv"))
    print(us_index.columns)
    print(us_index.head())
    
    # Economic Indicators: CPI, FedFundsRate, UnemploymentRate 
    cpi = read_csv('gold-etl', 'gold_data/CPI_data.csv')
    cpi.rename(columns={'date': 'Date', 'cpiaucsl': 'cpi'}, inplace=True)

    fed = read_csv('gold-etl', 'gold_data/FedFunds_data.csv')
    fed.rename(columns={'date': 'Date', 'fedfunds': 'fed_funds_rate'}, inplace=True)

    unemp = read_csv('gold-etl', 'gold_data/Unemployment_data.csv')
    unemp.rename(columns={'date': 'Date', 'unrate': 'unemployment_rate'}, inplace=True)
    
    # Step 2: stand for date merge (inner join) with gold, bitcoin, us_index
    df = pd.merge(gold[['Date', 'Close']], bitcoin[['Date', 'Close']], on='Date', suffixes=('_gold', '_btc'))
    print(df.tail())
    df = pd.merge(df, us_index[['Date', 'Close']], on='Date')
    print(df.tail())
    df.rename(columns={
        'Close_gold': 'gold_close',
        'Close_btc': 'bitcoin_close',
        'Close': 'us_index_close'
    }, inplace=True)
    print(df.tail())
    print(cpi.head())
    df = pd.merge(df, cpi, on='Date')
    print(df.head())
    df = pd.merge(df, fed, on='Date')
    print(df.head())
    df = pd.merge(df, unemp, on='Date')
    
    df = df.sort_values("Date").drop_duplicates(subset=["Date"])
    print(df["Date"].head(10))
    print(df["Date"].tail(10))
    
    # Step 3: Save the data to postgres
    print("df",df.head())
    conn = psycopg2.connect(
        dbname="airflow",
        user="airflow",
        password="airflow",
        host="postgres"
    )
    cur = conn.cursor()

    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO gold_model_data (
                date, gold_close, bitcoin_close, us_index_close, cpi, fed_funds_rate, unemployment_rate
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO NOTHING;
        """, (
            row['Date'], row['gold_close'], row['bitcoin_close'],
            row['us_index_close'], row['cpi'], row['fed_funds_rate'],
            row['unemployment_rate']
        ))

    conn.commit()
    cur.close()
    conn.close()


# Define the PythonOperator task
extract_task = PythonOperator(
    task_id='extract_gold_bitcoin_us_index',
    python_callable=extract_data,
    dag=dag,  
)

extract_econ_task = PythonOperator(
    task_id='extract_economic_indicators',
    python_callable=extract_economic_indicators,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='upload_to_minio',
    python_callable=upload_to_minio,
    dag=dag,
)

insert_model_data_task = PythonOperator(
    task_id='insert_gold_model_data',
    python_callable=insert_gold_model_data,
    dag=dag,
)
# Define the task dependencies
extract_task >> extract_econ_task >> upload_task >> insert_model_data_task