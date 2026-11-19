import http from "k6/http";
import { check } from "k6";

const TARGET_URL = __ENV.TARGET_URL || "http://127.0.0.1:8000";
const API_KEY = __ENV.API_KEY || "bench-secret-key-change-me";

const OOD_TEXTS = [
  "Hom nay troi mua to qua khong biet co di hoc duoc khong nua",
  "xkq9281 !! ## ?? zzzqwertyuiopasdfghjkl 00112233445566778899",
  "ok",
  "Sản phẩm giao hàng hơi chậm nhưng đóng gói rất cẩn thận và chắc chắn",
  "1234567890 abcdefghijklmnopqrstuvwxyz ".repeat(10).trim()
];

export const options = {
  vus: 5,
  iterations: 220
};

export default function () {
  const text = OOD_TEXTS[__ITER % OOD_TEXTS.length];
  const res = http.post(`${TARGET_URL}/v1/predict`, JSON.stringify({ texts: [text] }), {
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY
    }
  });
  check(res, { "status is 200": (r) => r.status === 200 });
}

