import os
from src.eda import EDAAutomator

def test_eda_report_generation():
    processed_data = "data/processed/heart_cleaned.csv"
    if not os.path.exists(processed_data):
        pytest.skip("Processed data missing")
        
    eda = EDAAutomator(processed_data)
    eda.run_full_report()
    
    assert os.path.exists("reports/class_distribution.png")
    assert os.path.exists("reports/correlation_heatmap.png")