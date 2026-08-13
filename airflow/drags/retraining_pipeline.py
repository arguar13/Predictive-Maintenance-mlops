from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_drift():
    """Aquí se consultaría la API de Prometheus/Evidently para evaluar si hay degradación."""
    # Simulación de verificación de Drift
    concept_drift_detected = True  # En producción, esto resulta de un request a Prometheus
    if not concept_drift_detected:
        raise ValueError("El modelo está saludable, no se requiere reentrenamiento.")

with DAG(
    'model_retraining_pipeline',
    default_args=default_args,
    schedule_interval='@weekly',
    catchup=False
) as dag:

    # 1. Verificar degradación
    drift_check = PythonOperator(
        task_id='check_concept_drift',
        python_callable=check_drift
    )

    # 2. Descargar nuevos datos de DVC
    pull_data = BashOperator(
        task_id='pull_new_data_dvc',
        bash_command='cd /opt/airflow && dvc pull'
    )

    # 3. Entrenar el nuevo FCN Baseline
    retrain_model = BashOperator(
        task_id='retrain_fcn_model',
        bash_command='python /opt/airflow/src/train.py'
    )

    # 4. Trigger CI/CD Pipeline (GitHub Actions webhook)
    trigger_cicd = BashOperator(
        task_id='trigger_github_actions',
        bash_command="""
        curl -X POST -H "Accept: application/vnd.github.v3+json" \
        -H "Authorization: token $GITHUB_TOKEN" \
        https://api.github.com/repos/tu-usuario/tu-repo/dispatches \
        -d '{"event_type": "model_retrained"}'
        """
    )

    drift_check >> pull_data >> retrain_model >> trigger_cicd