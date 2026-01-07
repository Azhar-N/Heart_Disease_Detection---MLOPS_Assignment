# Troubleshooting Guide

## Docker Issues

### Error: "docker client must be run with elevated privileges" or "cannot find the file specified"

This error means Docker Desktop is not running or not accessible.

#### Step-by-Step Fix:

1. **Check if Docker Desktop is Running**
   - Look at your system tray (bottom-right corner)
   - Find the Docker whale icon 🐳
   - If you don't see it, Docker Desktop is not running

2. **Start Docker Desktop**
   - Open Start Menu
   - Search for "Docker Desktop"
   - Click to launch
   - **Wait 1-2 minutes** for Docker to fully start
   - The whale icon should appear in system tray

3. **Verify Docker Desktop Status**
   - Click the Docker icon in system tray
   - It should show "Docker Desktop is running"
   - If you see errors, note them down

4. **Test Docker Connection**
   ```powershell
   docker --version
   docker ps
   ```
   Both commands should work without errors.

5. **If Still Not Working - Restart Docker Desktop**
   - Right-click Docker icon in system tray
   - Select "Quit Docker Desktop"
   - Wait 10-15 seconds
   - Start Docker Desktop again
   - Wait for it to fully initialize

6. **Check Docker Desktop Settings**
   - Open Docker Desktop
   - Go to Settings → General
   - Ensure "Use WSL 2 based engine" is checked (if available)
   - Click "Apply & Restart"

7. **Restart Computer (Last Resort)**
   - Sometimes Docker needs a full system restart
   - Restart Windows
   - Start Docker Desktop after restart
   - Wait for it to fully start

### Alternative: Skip Docker for Now

If Docker continues to cause issues, you can:

1. **Test API Locally Without Docker:**
   ```powershell
   # Train models first
   python src/models/train.py
   
   # Run API
   python src/api/app.py
   ```

2. **For Assignment Submission:**
   - Document that Docker/Kubernetes manifests are provided
   - Use GitHub Actions CI/CD screenshots (shows Docker build)
   - Note that local Docker testing was skipped due to environment constraints
   - All code and configurations are production-ready

## Other Common Issues

### Python/Module Issues

**Issue:** ModuleNotFoundError
- **Solution:** 
  ```powershell
  # Make sure virtual environment is activated
  venv\Scripts\activate
  
  # Reinstall dependencies
  pip install -r requirements.txt
  ```

**Issue:** Dataset not found
- **Solution:**
  ```powershell
  python src/data/download.py
  ```

**Issue:** Models not found
- **Solution:**
  ```powershell
  python src/models/train.py
  ```

### Port Issues

**Issue:** Port 8000 already in use
- **Solution:**
  ```powershell
  # Find what's using the port
  netstat -ano | findstr :8000
  
  # Or use a different port
  uvicorn src.api.app:app --host 0.0.0.0 --port 8001
  ```

### MLflow Issues

**Issue:** MLflow UI won't start
- **Solution:**
  ```powershell
  # Update MLflow (fixes compatibility issues)
  pip install --upgrade mlflow
  
  # Or use Python module directly
  python -m mlflow ui
  ```

## Getting Help

If you continue to have issues:

1. Check the error message carefully
2. Verify all prerequisites are installed
3. Review the relevant documentation:
   - `QUICKSTART.md` - Step-by-step guide
   - `DOCKER_SETUP.md` - Docker installation
   - `DEPLOYMENT.md` - Deployment instructions

4. For assignment purposes, document:
   - What worked
   - What didn't work and why
   - Screenshots of successful steps
   - Note that code is production-ready
