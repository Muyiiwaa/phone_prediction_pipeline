# Phone Price Prediction Pipeline

This project implements a full-stack machine learning pipeline to predict phone prices. It utilizes ZenML for orchestrating MLOps workflows, Weights & Biases (WandB) for experiment tracking, FastAPI for serving the model as an API, and Logfire for logging and system monitoring.

## Features

*   **End-to-End ML Pipeline:** Reproducible pipeline for data ingestion, preprocessing, model training, and evaluation using ZenML.
*   **Experiment Tracking:** Comprehensive tracking of experiments, metrics, and model artifacts with Weights & Biases.
*   **API Deployment:** A FastAPI application to serve the trained model for real-time predictions.
*   **Monitoring & Logging:** Centralized logging and system monitoring via Logfire.

## Tech Stack

*   **Python**
*   **ZenML:** For MLOps pipeline orchestration.
*   **Weights & Biases (WandB):** For experiment tracking and visualization.
*   **FastAPI:** For building the prediction API.
*   **Logfire:** For application logging and system monitoring.
*   (Other libraries as specified in `requirements.txt`)

## Prerequisites

*   Python 3.x (3.12) max. ZenML does not support 3.13
*   Git
*   A Weights & Biases account ([wandb.ai](https://wandb.ai)) to get an API key.
*  A Logfire account/setup for viewing logs.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Muyiiwaa/phone_prediction_pipeline
    cd phone_prediction_pipeline
    ```

2.  **Create and activate a virtual environment:**
    *   Create the environment:
        ```bash
        python -m venv phone_env
        ```
    *   Activate the environment:
        *   Windows:
            ```bash
            phone_env\Scripts\activate
            ```
        *   Linux/macOS:
            ```bash
            source phone_env/bin/activate
            ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Initialize ZenML:**
    This will set up ZenML for your project in the current directory.
    ```bash
    zenml init
    ```

5.  **Login to Weights & Biases:**
    *   Go to [wandb.ai](https://wandb.ai) and retrieve your API key.
    *   Run the following command and paste your API key when prompted:
        ```bash
        wandb login
        ```

## Running the ML Pipeline

1.  **Execute the ZenML pipeline:**
    This script will run the defined machine learning pipeline.
    ```bash
    python run_pipeline.py
    ```

2.  **Access the ZenML dashboard (local):**
    To view the pipeline runs, artifacts, and metadata locally:
    ```bash
    zenml login --local
    ```
    ZenML will provide a link (usually `http://127.0.0.1:8237` or similar) in your terminal. Open this link in your browser to inspect the pipeline.

3.  **Track experiments on Weights & Biases:**
    During the pipeline execution, links to your WandB project and specific runs will be printed in the console. You can also navigate to your [wandb.ai](https://wandb.ai) dashboard to find the experiments associated with this project.

## Running the FastAPI Application

1.  **Start the FastAPI server:**
    This command will start the API service, typically on `http://127.0.0.1:8000`.
    ```bash
    python main.py
    ```

2.  **Access Swagger UI / API Documentation:**
    Open your web browser and navigate to:
    ```
    http://localhost:8000/docs
    ```
    (Or `http://127.0.0.1:8000/docs` if `localhost:8000/docs` doesn't resolve directly to the correct port for your setup).
    Here you can interact with the API endpoints.

## Logging and Monitoring

*   **Logfire:** Check your application logs and monitor system performance at the Logfire link provided for your setup.
