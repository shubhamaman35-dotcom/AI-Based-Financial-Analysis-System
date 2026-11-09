"""
System Launcher & CLI Runner
One-click CLI runner for Streamlit Dashboard, FastAPI Backend, and Automated Tests.
"""

import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="AI Financial Intelligence System Launcher")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit Interactive Dashboard")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI REST Backend")
    parser.add_argument("--test", action="store_true", help="Run automated test suite")
    parser.add_argument("--port", type=int, default=8501, help="Port to run on")

    args = parser.parse_args()

    # Default to dashboard if no args provided
    if not args.api and not args.test:
        args.dashboard = True

    if args.dashboard:
        print(f"🚀 Launching Quantum Intelligence Streamlit Dashboard on port {args.port}...")
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(args.port)]
        subprocess.run(cmd)

    elif args.api:
        port = 8000 if args.port == 8501 else args.port
        print(f"⚡ Launching Quantum Intelligence FastAPI Backend on http://localhost:{port}...")
        cmd = [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", str(port), "--reload"]
        subprocess.run(cmd)

    elif args.test:
        print("🧪 Running Automated System Test Suite...")
        cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
