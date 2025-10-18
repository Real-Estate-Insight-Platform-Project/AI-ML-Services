from google.cloud import bigquery

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