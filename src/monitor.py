import pandas as pd
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.config_loader import load_config

logger = logging.getLogger(__name__)

def generate_drift_report(reference_data_path: str, current_data_path: str, output_path: str = "dashboard/drift_report.html"):
    """
    Generates a data drift report comparing reference data (training) against current data (production).
    """
    try:
        reference_data = pd.read_csv(reference_data_path)
        current_data = pd.read_csv(current_data_path)
        
        # Initialize Evidently Report with Data Drift Preset
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=reference_data, current_data=current_data)
        
        # Save as HTML for the Dashboard
        drift_report.save_html(output_path)
        logger.info(f"Data drift report successfully generated at {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate data drift report: {e}")
        raise

if __name__ == "__main__":
    config = load_config()
    # In production, 'current_data' would be fetched from a live database logging API requests.
    # Here we use placeholders for demonstration purposes.
    generate_drift_report(
        reference_data_path=config["data"]["processed_data_path"],
        current_data_path="data/recent_production_data.csv"
    )