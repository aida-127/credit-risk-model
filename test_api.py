# test_api.py
import requests
import json
import time

def test_api():
    print("=== Testing Credit Risk API ===\n")
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health Check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        print("   ✅ Health check passed")
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to API. Is Docker running?")
        print("   Run: docker-compose up -d")
        return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Model Info
    print("\n2. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model-info", timeout=5)
        print(f"   Status Code: {response.status_code}")
        data = response.json()
        print(f"   Model Type: {data.get('model_type')}")
        print(f"   Features: {', '.join(data.get('feature_names', []))}")
        print("   ✅ Model info retrieved")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Single Prediction
    print("\n3. Testing single prediction...")
    test_data = {
        "recency": 45.5,
        "frequency": 12.0,
        "total_amount": 50000.0,
        "avg_amount": 4166.67,
        "std_amount": 1500.0,
        "amount_variability": 0.36,
        "amount_range": 3000.0
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            timeout=10
        )
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Customer ID: {result.get('customer_id')}")
            print(f"   Risk Probability: {result.get('risk_probability')}")
            print(f"   Risk Class: {result.get('risk_class')}")
            print(f"   Risk Level: {result.get('risk_level')}")
            print("   ✅ Prediction successful!")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Batch Prediction
    print("\n4. Testing batch prediction...")
    batch_data = [
        {
            "recency": 45.5,
            "frequency": 12.0,
            "total_amount": 50000.0,
            "avg_amount": 4166.67,
            "std_amount": 1500.0,
            "amount_variability": 0.36,
            "amount_range": 3000.0
        },
        {
            "recency": 10.0,
            "frequency": 30.0,
            "total_amount": 150000.0,
            "avg_amount": 5000.0,
            "std_amount": 800.0,
            "amount_variability": 0.16,
            "amount_range": 2000.0
        }
    ]
    
    try:
        response = requests.post(
            f"{base_url}/predict-batch",
            json=batch_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Total Customers: {result.get('total_customers')}")
            print(f"   High Risk Count: {result.get('high_risk_count')}")
            print("   ✅ Batch prediction successful!")
        else:
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*50)
    print("🎉 API Testing Complete!")
    print(f"Open API docs: {base_url}/docs")

if __name__ == "__main__":
    test_api()