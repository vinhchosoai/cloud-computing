import http from "k6/http";
import { check } from "k6";
import { Trend } from "k6/metrics";

const BATCH_SIZE = parseInt(__ENV.BATCH_SIZE || "1", 10);
const TARGET_URL = __ENV.TARGET_URL || "http://127.0.0.1:8000";
const API_KEY = __ENV.API_KEY || "bench-secret-key-change-me";
const VUS = parseInt(__ENV.VUS || "10", 10);
const RAMP = __ENV.RAMP || "10s";
const DURATION = __ENV.DURATION || "2m";
const DOWN = __ENV.DOWN || "5s";

const serverLatencyTrend = new Trend("server_latency_ms");

const SAMPLE_TEXTS = [
  "An absolute masterpiece with brilliant performances and unforgettable direction.",
  "Tedious, predictable, and a complete waste of two hours.",
  "A charming, heartfelt comedy that leaves you smiling.",
  "The plot makes no sense and the dialogue feels completely artificial.",
  "Stunning visuals paired with a deeply moving soundtrack.",
  "Boring from start to finish with zero character development.",
  "Smart, witty, and surprisingly emotional.",
  "I wanted to walk out of the theater halfway through."
];

export const options = {
  stages: [
    { duration: RAMP, target: VUS },
    { duration: DURATION, target: VUS },
    { duration: DOWN, target: 0 }
  ],
  thresholds: {
    http_req_failed: ["rate<0.05"]
  }
};

export default function () {
  const texts = [];
  for (let i = 0; i < BATCH_SIZE; i++) {
    texts.push(SAMPLE_TEXTS[(i + __ITER + __VU) % SAMPLE_TEXTS.length]);
  }
  const payload = JSON.stringify({ texts: texts });
  const params = {
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY
    },
    timeout: "30s"
  };

  const res = http.post(`${TARGET_URL}/v1/predict`, payload, params);
  check(res, {
    "status is 200": (r) => r.status === 200
  });
  if (res.status === 200) {
    try {
      const body = res.json();
      if (body && typeof body.latency_ms === "number") {
        serverLatencyTrend.add(body.latency_ms);
      }
    } catch (_) {}
  }
}

