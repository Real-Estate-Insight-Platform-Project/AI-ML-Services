FROM apache/airflow:3.0.6

# Switch back to airflow user
USER airflow

# Copy requirements file into the image
COPY requirements.txt .

# Install additional Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
