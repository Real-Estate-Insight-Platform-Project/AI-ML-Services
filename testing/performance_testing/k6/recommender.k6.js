import http from "k6/http";
import { sleep, check } from "k6";

export const options = {
  scenarios: {
    // Quick validation test - minimal load
    baseline: { 
      executor: "shared-iterations", 
      vus: 2, 
      iterations: 10, 
      maxDuration: "30s" 
    },
    // Light ramp test
    ramp: {
      executor: "ramping-vus",
      startVUs: 1,
      stages: [
        { duration: "15s", target: 3 },
        { duration: "15s", target: 3 },
        { duration: "10s", target: 0 }
      ]
    }
  },
  thresholds: {
    http_req_failed: ["rate<0.05"],           // <5% errors (more lenient)
    http_req_duration: ["p(95)<2000"],        // p95 < 2 seconds (very lenient)
    http_req_duration: ["avg<1000"]           // average < 1 second
  }
};

const BASE = __ENV.RECOMMENDER_BASE;

function pick(arr) { 
  return arr[Math.floor(Math.random() * arr.length)]; 
}

export default function () {
  // Create proper JSON payload for POST /recommend endpoint
  const payload = {
    top_k: pick([5, 10, 15]),
    locations: pick([
      ["Miami, Florida"],
      ["Orlando, Florida"], 
      ["Tampa, Florida"],
      ["Miami, Florida", "Orlando, Florida"],
      []  // empty for testing all locations
    ]),
    property_types: pick([
      ["Single Family"],
      ["Condo"],
      ["Single Family", "Condo"],
      ["Single Family", "Condo", "Townhouse"],
      []  // empty for all property types
    ]),
    price_min: pick([200000, 300000, 400000, null]),
    price_max: pick([600000, 800000, 1000000, null]),
    min_rating: pick([4.0, 4.5, null]),
    min_reviews: pick([50, 100, null]),
    require_phone: pick([true, false])
  };

  // Remove null values to test different request variations
  Object.keys(payload).forEach(key => {
    if (payload[key] === null) {
      delete payload[key];
    }
  });

  // Ensure price_min and price_max are both provided or both omitted
  if (payload.hasOwnProperty('price_min') && !payload.hasOwnProperty('price_max')) {
    payload.price_max = payload.price_min * 2;
  } else if (payload.hasOwnProperty('price_max') && !payload.hasOwnProperty('price_min')) {
    payload.price_min = Math.floor(payload.price_max * 0.5);
  }
  
  const url = `${BASE}/recommend`;
  const res = http.post(url, JSON.stringify(payload), {
    headers: { 
      'Content-Type': 'application/json' 
    },
    tags: { svc: "recommender" }
  });
  
  check(res, {
    "200": (r) => r.status === 200,
    "valid response structure": (r) => {
      try { 
        const data = r.json();
        return (
          data.hasOwnProperty('total_matches') &&
          data.hasOwnProperty('returned') &&
          data.hasOwnProperty('results') &&
          Array.isArray(data.results)
        ); 
      } catch { 
        return false; 
      }
    },
    "returned count matches": (r) => {
      try {
        const data = r.json();
        return data.returned === data.results.length;
      } catch {
        return false;
      }
    }
  });
  
  sleep(1);
}