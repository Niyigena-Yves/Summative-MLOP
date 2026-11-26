"""Flood the prediction endpoint with requests.

Run (headless, results saved as CSV):
    locust -f locust/locustfile.py --host http://localhost:8080 \
        --users 100 --spawn-rate 10 --run-time 60s --headless \
        --csv locust/results/c3_u100
"""
import os
import random

from locust import HttpUser, between, task

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(ROOT, "data", "test")
IMAGES = []
for cls in sorted(os.listdir(TEST_DIR)):
    d = os.path.join(TEST_DIR, cls)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d))[:10]:
            with open(os.path.join(d, f), "rb") as fh:
                IMAGES.append((f, fh.read()))


class FarmerUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        self.client.post("/predict", files={"file": (name, data, "image/jpeg")}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
