FROM apache/airflow:3.0.6

# Install system dependencies required by LightGBM
USER root
RUN apt-get update && apt-get install -y libgomp1 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy requirements file into the image
COPY requirements.txt .

# Install additional Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
