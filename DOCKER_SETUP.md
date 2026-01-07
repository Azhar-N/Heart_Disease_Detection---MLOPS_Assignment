# Docker Setup Guide

## Installing Docker Desktop on Windows

Docker is **optional** for local development and testing. You can run the API directly using Python (see QUICKSTART.md Step 7). However, Docker is required for:
- Containerization demonstration
- Kubernetes deployment
- Production-like environment testing

### Installation Steps

1. **Download Docker Desktop**
   - Visit: https://www.docker.com/products/docker-desktop/
   - Click "Download for Windows"
   - Choose the appropriate version (usually Docker Desktop for Windows)

2. **Install Docker Desktop**
   - Run the installer (Docker Desktop Installer.exe)
   - Follow the installation wizard
   - **Important:** You may need to enable WSL 2 (Windows Subsystem for Linux) if prompted
   - Restart your computer if required

3. **Start Docker Desktop**
   - Launch Docker Desktop from Start Menu
   - Wait for Docker to start (you'll see a whale icon in the system tray)
   - The Docker Desktop dashboard will open

4. **Verify Installation**
   Open PowerShell and run:
   ```powershell
   docker --version
   docker ps
   ```
   Both commands should work without errors.

### Common Issues

#### Issue: "Docker command not recognized"
- **Solution:** Restart your computer after installation
- Make sure Docker Desktop is running (check system tray)
- Restart PowerShell/terminal after Docker Desktop starts

#### Issue: "WSL 2 installation is incomplete"
- **Solution:** Install WSL 2:
  1. Open PowerShell as Administrator
  2. Run: `wsl --install`
  3. Restart your computer
  4. Try starting Docker Desktop again

#### Issue: "Docker Desktop won't start"
- **Solution:** 
  - Check Windows updates
  - Ensure virtualization is enabled in BIOS
  - Check Docker Desktop logs in Settings → Troubleshoot

#### Issue: "error during connect: docker client must be run with elevated privileges"
- **Solution:** 
  1. **Make sure Docker Desktop is running:**
     - Look for Docker icon in system tray (whale icon)
     - If not visible, start Docker Desktop from Start Menu
     - Wait 1-2 minutes for Docker to fully start
     - The icon should be steady (not animated) when ready
  
  2. **Check Docker Desktop status:**
     - Open Docker Desktop application
     - Look for "Docker Desktop is running" message
     - Check if there are any error messages
  
  3. **Restart Docker Desktop:**
     - Right-click Docker icon in system tray
     - Select "Quit Docker Desktop"
     - Wait 10 seconds
     - Start Docker Desktop again from Start Menu
     - Wait for it to fully start
  
  4. **Run PowerShell as Administrator (if needed):**
     - Right-click PowerShell
     - Select "Run as Administrator"
     - Try Docker commands again
  
  5. **Verify Docker is accessible:**
     ```powershell
     docker --version
     docker ps
     ```
     Both should work without errors.

### Using Docker

Once Docker is installed and running:

1. **Build the image:**
   ```powershell
   docker build -t heart-disease-api:latest -f docker/Dockerfile .
   ```

2. **Run the container:**
   ```powershell
   docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest
   ```

3. **Check if container is running:**
   ```powershell
   docker ps
   ```

4. **View logs:**
   ```powershell
   docker logs heart-disease-api
   ```

5. **Stop the container:**
   ```powershell
   docker stop heart-disease-api
   docker rm heart-disease-api
   ```

### Alternative: Test Without Docker

If you don't want to install Docker, you can still complete most of the assignment:

1. ✅ Data Acquisition & EDA
2. ✅ Model Training
3. ✅ Experiment Tracking (MLflow)
4. ✅ API Testing (run locally with Python)
5. ✅ CI/CD Pipeline (GitHub Actions)
6. ❌ Docker Containerization (requires Docker)
7. ❌ Kubernetes Deployment (requires Docker/Kubernetes)

For the assignment, you can:
- Document that Docker/Kubernetes deployment was prepared but not tested locally
- Use screenshots from GitHub Actions CI/CD pipeline showing Docker build
- Note that deployment manifests are provided and ready for cloud deployment
