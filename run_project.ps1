Write-Host "🚀 Starting FastAPI..."

$env:API_URL="http://localhost:8000"
$env:BUSINESS_THRESHOLD="0.40"
$env:TOP_K_RETURN="50"
$env:MAX_BATCH_ROWS="50000"

Start-Process powershell -ArgumentList "uvicorn src.api.main:app --reload"

Start-Sleep -Seconds 3

Write-Host "🌐 Starting Streamlit..."
Start-Process powershell -ArgumentList "streamlit run src/app_web.py"

