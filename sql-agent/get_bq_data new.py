import os
from google.cloud import bigquery

os.environ['GOOGLE_APPLICATION_CREDENTIALS' ] = 'service_keys.json'

client = bigquery. Client()

def get_bq_data(client, table_name):

    sql = f"""
    SELECT
    *
    FROM
    `fourth-webbing-474805-j5.real_estate_market.{table_name}`;
    """
    job = client.query(sql)
    df = job.to_dataframe()

    return df