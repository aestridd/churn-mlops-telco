Write-Host "🚀 Starting FastAPI..."
Start-Process powershell -ArgumentList "uvicorn src.api.main:app --reload"

Start-Sleep -Seconds 3

Write-Host "🌐 Starting Streamlit..."
Start-Process powershell -ArgumentList "streamlit run src/app_web.py"
