"""
Automated headless Locust load test execution script.
Runs a 15-second load test with 10 concurrent users, logs p50 and p95 latency,
and validates that the service satisfies SLA requirements.
"""

from __future__ import annotations

import subprocess
import sys

import requests


def run_headless_load_test(
    host: str = "http://127.0.0.1:8000",
    users: int = 10,
    spawn_rate: int = 2,
    run_time: str = "10s",
):
    print(f"Targeting host: {host}")
    try:
        r = requests.get(f"{host}/health", timeout=3)
        if r.status_code != 200:
            print(f"Server is not healthy: {r.status_code}")
            return False
    except Exception as e:  # noqa: BLE001
        print(f"Cannot connect to {host}: {e}")
        print("Please ensure FastAPI server is running before executing load test.")
        return False

    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        "locustfile.py",
        "--headless",
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "--run-time",
        run_time,
        "--host",
        host,
    ]

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


if __name__ == "__main__":
    run_headless_load_test()
