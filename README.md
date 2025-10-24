# AI-ML-Services

A modular, service-oriented platform for real estate insights, integrating forecasting, recommendation, and NLP-based interaction through containerized microservices.

## Services

### ETL Data Pipeline
- **Description**: Automated monthly data pipeline using Apache Airflow for extracting housing data from Realtor.com, transforming it, and loading into BigQuery while retraining ML models.
- **How to run**:
  ```bash
  cd ETL
  docker-compose up
  ```

### Forecasting Engine
- **Description**: Stacking ensemble model for predicting real estate metrics at state and county levels using Darts framework.
- **How to run**:
  ```bash
  cd forecasting_engine
  pip install -r requirements.txt
  jupyter notebook
  # Open and run modeling.ipynb or preprocessing.ipynb
  ```

### Agent Finder
- **Description**: ML-powered real estate agent recommendation system using text embeddings and Bayesian calibration.
- **How to run**:
  ```bash
  cd agent_finder
  pip install -r requirements.txt
  python main.py
  ```

### Property Recommender
- **Description**: Property recommendation service.
- **How to run**:
  ```bash
  cd property_recommender
  pip install -r requirements.txt
  python main.py
  ```

### SQL Agent
- **Description**: SQL-based agent for data queries.
- **How to run**:
  ```bash
  cd sql-agent
  pip install -r requirements.txt
  python main.py
  ```

## System Design

The Smart Real Estate Insight Platform was designed using a modular, service-oriented architecture that integrates multiple data-driven components under a unified web interface. The architecture emphasizes scalability, maintainability, and interoperability between analytical services such as forecasting, recommendation, and NLP-based interaction. The system follows a three-tier architecture consisting of a data layer, application/analytics layer, and presentation layer, each operating within containerized environments to ensure modular deployment and easy version control through MLOps tools such as MLflow and Docker.

At the data layer, the ETL pipeline continuously extracts monthly housing, weather, and geo-contextual datasets from sources such as Realtor.com and NOAA. These are transformed using Python and Pandas scripts and loaded into Google Big Query or Supabase. This layer ensures data integrity and provides a consistent, query-ready structure for the analytical modules.

The application layer hosts the platform's core services, Forecasting Engine, Agentic NLP Assistant, Agent Finder, Property Finder, Risk Assessment Module, and Investment Analysis Engine. These microservices communicate through REST APIs built with FastAPI, enabling asynchronous interaction and seamless integration with the frontend. Each module operates independently but shares access to centralized metadata and logs for monitoring.

The presentation layer, implemented using Next.js, delivers a dynamic and responsive dashboard for buyers, investors, and administrators. This layer visualizes market trends, property recommendations, and forecasts using interactive charts while the agentic NLP assistant enables natural-language exploration of insights.

## Project Abstract Architecture
![Project Abstract Architecture](Images/DSE%20Project%20Abstract%20Architecture.png)


## ETL Data Pipeline

This automated data pipeline, orchestrated by Apache Airflow, runs monthly to extract the latest housing market data from Realtor.com. The data is then transformed, preprocessed, and modified for two parallel purposes: it's loaded into Google Cloud BigQuery as aggregated historical data, and it's simultaneously used to retrain machine learning models via MLflow. Finally, the newly generated predictions from these models are also loaded back into BigQuery for use by the application.

## ETL Pipeline
![ETL Pipeline](Images/ETL%20Pipeline.png)

## Forecasting Engine

This forecasting engine uses a stacking ensemble (meta-model) architecture to predict five key real estate metrics, such as median listing price and days on market, at both the state and county levels. It is built using the Darts framework, which efficiently trains multiple baseline models on a rich dataset of 30+ past covariates (like demographic and market factors) and their lag features. The predictions from these base models are then fed into a final "stacking layer" which combines them to produce a single, more robust and accurate forecast.

## ML model Architecture
![Forecasting Engine](Images/ML%20model%20architecture.png)