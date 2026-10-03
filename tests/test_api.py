import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.index import app
import base64

client = TestClient(app)

# Basic Auth credentials based on defaults
auth_string = "admin:admin123"
b64_auth = base64.b64encode(auth_string.encode()).decode()
headers = {"Authorization": f"Basic {b64_auth}"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_unauthorized():
    # Should fail without auth headers
    payload = {
        "customer_id": "CUST-001",
        "revenue": 1000.0,
        "monthly_charges": 85.5,
        "usage_frequency": 20,
        "complaints_count": 0,
        "payment_delays": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 401

def test_predict_endpoint_authorized():
    payload = {
        "customer_id": "CUST-001",
        "revenue": 1000.0,
        "monthly_charges": 85.5,
        "usage_frequency": 20,
        "complaints_count": 0,
        "payment_delays": 0
    }
    response = client.post("/predict", json=payload, headers=headers)
    # Could be 200 (success) or 503 (model not available) depending on state
    assert response.status_code in (200, 503)
    
    if response.status_code == 200:
        data = response.json()
        assert "customer_id" in data
        assert "churn_probability" in data
        assert "risk_bucket" in data
