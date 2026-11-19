import http from "k6/http";
import { check } from "k6";

const TARGET_URL = __ENV.TARGET_URL || "http://127.0.0.1:8000";
const API_KEY = __ENV.API_KEY || "bench-secret-key-change-me";

export const options = {
  stages: [
    { duration: "30s", target: 2 },
    { duration: "10s", target: 50 },
    { duration: "2m", target: 50 },
    { duration: "20s", target: 2 }
  ]
};

export default function () {
  const payload = JSON.stringify({
    texts: ["A remarkably entertaining film with strong pacing and sharp humor."]
  });
  const res = http.post(`${TARGET_URL}/v1/predict`, payload, {
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY
    },
    timeout: "30s"
  });
  check(res, { "status is 200": (r) => r.status === 200 });
}

