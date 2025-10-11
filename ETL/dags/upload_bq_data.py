from google.cloud import bigquery

def upload_bq_data(client, table_name, df, method):

    table_id = f"fourth-webbing-474805-j5.real_estate_market.{table_name}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=method,  # Options: WRITE_TRUNCATE, WRITE_APPEND, WRITE_EMPTY
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # Wait for the job to complete

    print(f"Loaded {job.output_rows} rows into {table_id}")

    return df