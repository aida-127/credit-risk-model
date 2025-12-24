# test_api.ps1
Write-Host "=== TASK 6 TEST ===" -ForegroundColor Green

# 1. Docker status
Write-Host "`n1. Docker Container:" -ForegroundColor Yellow
docker ps

# 2. Health check
Write-Host "`n2. API Health:" -ForegroundColor Yellow
curl.exe http://localhost:8000/health

# 3. Test prediction with Python (always works)
Write-Host "`n3. Testing Prediction with Python:" -ForegroundColor Yellow
python -c "
import requests
data = {
    'recency': 45.5,
    'frequency': 12.0,
    'total_amount': 50000.0,
    'avg_amount': 4166.67,
    'std_amount': 1500.0,
    'amount_variability': 0.36,
    'amount_range': 3000.0
}
try:
    response = requests.post('http://localhost:8000/predict', json=data)
    print('Status:', response.status_code)
    print('Result:', response.json())
except Exception as e:
    print('Error:', e)
"

Write-Host "`n=== TEST COMPLETE ===" -ForegroundColor Green