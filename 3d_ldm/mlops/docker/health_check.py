"""
Health check script for Docker container
Validates model availability and system health
"""

import sys
import json
import time
import requests
import subprocess
from pathlib import Path

def check_gpu():
    """Check GPU availability"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def check_model_files():
    """Check if model files are available"""
    model_path = Path("/app/models/3d_ldm_model.pth")
    return model_path.exists()

def check_api_server():
    """Check if API server is responding"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/app")
        free_gb = free / (1024**3)
        return free_gb > 1.0  # At least 1GB free
    except:
        return False

def check_memory():
    """Check available memory"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.available > 1024*1024*1024  # At least 1GB available
    except:
        return False

def main():
    """Main health check function"""
    health_status = {
        "timestamp": time.time(),
        "status": "healthy",
        "checks": {}
    }
    
    # Perform health checks
    checks = {
        "gpu_available": check_gpu(),
        "model_files": check_model_files(),
        "disk_space": check_disk_space(),
        "memory": check_memory()
    }
    
    # Check API server if it should be running
    if Path("/tmp/api_server.pid").exists():
        checks["api_server"] = check_api_server()
    
    health_status["checks"] = checks
    
    # Determine overall health
    critical_checks = ["disk_space", "memory"]
    failed_critical = [check for check in critical_checks if not checks.get(check, False)]
    
    if failed_critical:
        health_status["status"] = "unhealthy"
        health_status["failed_checks"] = failed_critical
        print(json.dumps(health_status, indent=2))
        sys.exit(1)
    
    # Warnings for non-critical checks
    warnings = []
    if not checks.get("gpu_available", False):
        warnings.append("GPU not available - using CPU")
    if not checks.get("model_files", False):
        warnings.append("Model files not found - using dummy model")
    
    if warnings:
        health_status["warnings"] = warnings
    
    print(json.dumps(health_status, indent=2))
    sys.exit(0)

if __name__ == "__main__":
    main()