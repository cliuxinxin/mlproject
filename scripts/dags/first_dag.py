from datetime import datetime, timedelta
from data_utils import *

from airflow.decorators import dag, task


default_args = {
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

@dag(dag_id='update_data', 
     default_args=default_args, 
     start_date=datetime(2021, 10, 26), 
     schedule_interval='@daily')
def hello_world_etl():
    @task()
    def get_name():
        b_doccano_train_dev_update('tender')

greet_dag = hello_world_etl()