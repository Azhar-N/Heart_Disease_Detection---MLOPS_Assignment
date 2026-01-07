# PowerShell script to test the API endpoints

$API_URL = "http://localhost:8000"

Write-Host "Testing Heart Disease Prediction API" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green

# Health check
Write-Host "`n1. Health Check:" -ForegroundColor Yellow
$healthResponse = Invoke-RestMethod -Uri "$API_URL/health" -Method Get
$healthResponse | ConvertTo-Json

# Prediction
Write-Host "`n2. Prediction:" -ForegroundColor Yellow
$requestBody = Get-Content -Path "sample_request.json" -Raw
$predictionResponse = Invoke-RestMethod -Uri "$API_URL/predict" -Method Post -Body $requestBody -ContentType "application/json"
$predictionResponse | ConvertTo-Json

# Metrics
Write-Host "`n3. Metrics:" -ForegroundColor Yellow
$metricsResponse = Invoke-WebRequest -Uri "$API_URL/metrics" -Method Get
$metricsResponse.Content.Split("`n")[0..19] | ForEach-Object { Write-Host $_ }

Write-Host "`nTesting completed!" -ForegroundColor Green
