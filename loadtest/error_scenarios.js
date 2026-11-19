import http from "k6/http";
import { check } from "k6";

const TARGET_URL = __ENV.TARGET_URL || "http://127.0.0.1:8000";
const API_KEY = __ENV.API_KEY || "dev-secret-key-change-me";

export const options = {
  vus: 1,
  iterations: 1
};

export default function () {
  // 1. Missing API key -> 401
  const r1 = http.post(`${TARGET_URL}/v1/predict`, JSON.stringify({ texts: ["hello"] }), {
    headers: { "Content-Type": "application/json" }
  });
  check(r1, { "missing key -> 401": (r) => r.status === 401 });

  // 2. Invalid API key -> 403
  const r2 = http.post(`${TARGET_URL}/v1/predict`, JSON.stringify({ texts: ["hello"] }), {
    headers: { "Content-Type": "application/json", "X-API-Key": "invalid-key" }
  });
  check(r2, { "invalid key -> 403": (r) => r.status === 403 });

  // 3. Empty texts -> 422
  const r3 = http.post(`${TARGET_URL}/v1/predict`, JSON.stringify({ texts: [] }), {
    headers: { "Content-Type": "application/json", "X-API-Key": API_KEY }
  });
  check(r3, { "empty batch -> 422": (r) => r.status === 422 });

  // 4. Text > 512 chars -> 422
  const r4 = http.post(`${TARGET_URL}/v1/predict`, JSON.stringify({ texts: ["x".repeat(520)] }), {
    headers: { "Content-Type": "application/json", "X-API-Key": API_KEY }
  });
  check(r4, { "oversized text -> 422": (r) => r.status === 422 });

  // 5. Trigger rate limit -> 429
  let got429 = false;
  for (let i = 0; i < 70; i++) {
    const r = http.post(`${TARGET_URL}/v1/predict`, JSON.stringify({ texts: ["rate limit test"] }), {
      headers: { "Content-Type": "application/json", "X-API-Key": API_KEY }
    });
    if (r.status === 429) {
      got429 = true;
      break;
    }
  }
  check(got429, { "rate limit triggered 429": (v) => v === true });
}

