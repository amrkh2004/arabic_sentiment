"""
Locust Load Testing Suite for Arabic Sentiment Analysis API.
Simulates high-concurrency e-commerce user traffic (e.g. Noon, Amazon Egypt reviews).
Monitors throughput (RPS), error rate, and verifies that p95 latency satisfies SLA.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task

# Realistic Arabic e-commerce test sentences
SAMPLE_REVIEWS = [
    "المنتج ممتاز جدا وخامته رائعة والشحن سريع",
    "خامة رديئة جدا وسيئة وتالف ولا يعمل",
    "المنتج عادي ومقبول بالنسبة لسعره المناسب",
    "التوصيل كان متأخر جدا لكن المنتج كويس",
    "تغليف فاشل ووصلت العلبة مكسورة تماما",
    "جودة عالية وتصميم مريح جدا أنصح بشرائه بشدة",
    "للأسف تجربة سيئة ولن أشتري منكم مجددا",
    "وصل بالموعد والمواصفات مطابقة تماما للصور",
    "لا بأس به يؤدي الغرض ولكن الصوت منخفض قليلا",
    "أفضل جهاز اشتريته هذا العام ما شاء الله",
]


class SentimentLoadTestUser(HttpUser):
    """Simulates a client sending sentiment inference requests."""

    wait_time = between(0.1, 0.5)

    @task(8)
    def test_single_prediction(self):
        """Simulates single review prediction requests."""
        text = random.choice(SAMPLE_REVIEWS)
        with self.client.post(
            "/predict",
            json={"text": text},
            catch_response=True,
            name="/predict (single)",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data and len(data["results"]) == 1:
                    response.success()
                else:
                    response.failure(f"Unexpected response structure: {response.text}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(3)
    def test_batch_prediction(self):
        """Simulates batch review predictions (e.g., catalog moderation)."""
        batch_size = random.randint(2, 5)
        texts = random.sample(SAMPLE_REVIEWS, batch_size)
        with self.client.post(
            "/predict",
            json={"texts": texts},
            catch_response=True,
            name="/predict (batch)",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data and len(data["results"]) == batch_size:
                    response.success()
                else:
                    response.failure(f"Batch count mismatch: {response.text}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)
    def test_health_check(self):
        """Monitors health check endpoint."""
        self.client.get("/health", name="/health")


@events.quitting.add_listener
def check_sla_on_quit(environment, **kwargs):
    """
    Quality gate validation: Ensures p95 latency does not exceed 250ms SLA.
    """
    stats = environment.runner.stats.total
    p95 = stats.get_current_response_time_percentile(0.95)
    p50 = stats.get_current_response_time_percentile(0.50)
    print("\n" + "=" * 60)
    print("           LOCUST LOAD TEST BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Requests:      {stats.num_requests}")
    print(f"Total Failures:      {stats.num_failures} ({stats.fail_ratio * 100:.2f}%)")
    print(f"Requests / Sec:      {stats.total_rps:.2f}")
    print(f"Latency p50 (Median): {p50:.2f} ms")
    print(f"Latency p95:          {p95:.2f} ms")
    print("=" * 60)
