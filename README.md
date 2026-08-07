# Hotel Booking Market Segment Classification Pipeline

A production-ready MLOps project designed to automate data preprocessing, model training, artifact tracking, and API serving for predicting hotel booking market segments.

---

## Project Structure

```text
hotel_booking_mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processing.py
│   ├── logger.py
│   └── train.py
├── .gitignore
├── config.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## System Requirements

- Python 3.10 or higher
- Docker (optional, for containerized deployment)
- Git

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hotel-booking-mlops.git
cd hotel-booking-mlops
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the Dataset

Place your raw dataset inside the `data/` directory.

**Required file path:**

```text
data/hotel_bookings.csv
```

---

## Configuration

System settings, path definitions, and hyperparameter configurations are managed through `config.yaml`.

Modify this file to adjust tracking URIs, train/test split settings, or model parameters.

```yaml
project:
  name: "Hotel_Booking_Market_Segment"
  version: "1.0.0"

paths:
  data_path: "data/hotel_bookings.csv"
  model_dir: "models/"

mlflow:
  experiment_name: "market_segment_classification"
  tracking_uri: "sqlite:///mlflow.db"

model_params:
  test_size: 0.2
  random_state: 42
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
```

---

## Model Training and Experiment Tracking

Run the training pipeline to execute:

- Data cleaning
- Feature preprocessing
- Class balancing using SMOTE
- XGBoost model training
- MLflow experiment logging

### Train the Model

```bash
python -m src.train
```

### Launch MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the following URL in your browser:

```text
http://127.0.0.1:5000
```

---

## API Serving

Start the FastAPI application locally using Uvicorn.

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Documentation

Once the API is running, interactive documentation is available at:

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

### ReDoc

```text
http://127.0.0.1:8000/redoc
```

---

## Example Request

### Endpoint

```http
POST /predict
```

### Request Body

```json
{
  "hotel": "Resort Hotel",
  "lead_time": 342,
  "arrival_date_year": 2015,
  "arrival_date_week_number": 27,
  "arrival_date_day_of_month": 1,
  "stays_in_weekend_nights": 0,
  "stays_in_week_nights": 0,
  "adults": 2,
  "children": 0.0,
  "babies": 0,
  "meal": "BB",
  "country": "PRT",
  "distribution_channel": "Direct",
  "is_repeated_guest": 0,
  "previous_cancellations": 0,
  "previous_bookings_not_canceled": 0,
  "reserved_room_type": "C",
  "assigned_room_type": "C",
  "booking_changes": 3,
  "deposit_type": "No Deposit",
  "days_in_waiting_list": 0,
  "customer_type": "Transient",
  "adr": 0.0,
  "required_car_parking_spaces": 0,
  "total_of_special_requests": 0,
  "month": 7
}
```

---

## Containerization

Build and run the Docker container to ensure environment reproducibility across infrastructures.

### Build the Docker Image

```bash
docker build -t hotel-booking-mlops:latest .
```

### Run the Docker Container

```bash
docker run -p 8000:8000 hotel-booking-mlops:latest
```

---

## Continuous Integration (CI)

The repository includes a GitHub Actions workflow located at:

```text
.github/workflows/ci.yml
```

On every push or pull request to the `main` branch, the workflow automatically executes:

- Flake8 linting and syntax validation
- Static type checking with Mypy
- Automated unit testing with Pytest

---

## Technology Stack

- Python
- FastAPI
- Uvicorn
- XGBoost
- Scikit-learn
- SMOTE
- MLflow
- Docker
- GitHub Actions
- Pytest
- Mypy
- Flake8

---

## License

This project is licensed under the MIT License.