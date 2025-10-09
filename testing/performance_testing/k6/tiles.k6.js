import http from "k6/http";
import { check } from "k6";
import { SharedArray } from "k6/data";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.4/index.js";

export const options = {
  scenarios: {
    // Quick tile validation - light load
    tiles_test: { 
      executor: "shared-iterations", 
      vus: 3, 
      iterations: 15, 
      maxDuration: "45s" 
    }
  },
  thresholds: {
    http_req_failed: ["rate<0.1"],       // <10% errors (very lenient)
    http_req_duration: ["p(95)<1500"],   // p95 < 1.5 seconds
    http_req_duration: ["avg<800"]       // average < 800ms
  }
};

const BASE = __ENV.RISKMAP_BASE;

// Generate tile coordinates for different zoom levels
const tiles = new SharedArray("tiles", () => {
  const tileData = [];
  
  // World level (z=0-2)
  for (let z = 0; z <= 2; z++) {
    const max = Math.pow(2, z);
    for (let x = 0; x < max; x++) {
      for (let y = 0; y < max; y++) {
        tileData.push({ z, x, y, level: "world" });
      }
    }
  }
  
  // Regional level (z=3-7) - sample coordinates
  const regionalTiles = [
    { z: 5, x: 9, y: 12 },   // North America area
    { z: 6, x: 18, y: 24 },  // USA area
    { z: 7, x: 36, y: 49 }   // Southeast USA
  ];
  regionalTiles.forEach(tile => {
    tileData.push({ ...tile, level: "region" });
  });
  
  // City level (z=8-12) - Florida/Miami area
  const cityTiles = [
    { z: 10, x: 284, y: 410 },  // Miami-Dade area
    { z: 10, x: 305, y: 380 },  // Central Florida
    { z: 12, x: 1150, y: 1850 }, // Detailed city view
    { z: 11, x: 568, y: 820 }   // Tampa area
  ];
  cityTiles.forEach(tile => {
    tileData.push({ ...tile, level: "city" });
  });
  
  // High detail level (z=13-18) - for properties
  const detailTiles = [
    { z: 15, x: 9300, y: 15100 },
    { z: 14, x: 4650, y: 7550 },
    { z: 13, x: 2325, y: 3775 }
  ];
  detailTiles.forEach(tile => {
    tileData.push({ ...tile, level: "detail" });
  });
  
  return tileData;
});

function pick(arr) { 
  return arr[Math.floor(Math.random() * arr.length)]; 
}

export default function () {
  // Select a random tile coordinate
  const tile = pick(tiles);
  
  // Randomly choose between counties and properties tiles
  const tileType = pick(["counties", "properties"]);
  
  const url = `${BASE}/tiles/${tileType}/${tile.z}/${tile.x}/${tile.y}`;
  const res = http.get(url, { 
    tags: { 
      svc: "tiles", 
      type: tileType,
      z: tile.z.toString(),
      level: tile.level 
    }
  });
  
  check(res, {
    "success": r => r.status === 200 || r.status === 204, // 204 for empty tiles is valid
    "mvt_content_type": r => (r.headers["Content-Type"] || r.headers["content-type"] || "").includes("application/vnd.mapbox-vector-tile"),
    "has_cache_headers": r => r.headers["Cache-Control"] || r.headers["cache-control"],
    "valid_mvt_response": r => {
      // 200 should have content, 204 should be empty
      if (r.status === 200) {
        return r.body && r.body.length && r.body.length > 0;
      } else if (r.status === 204) {
        return !r.body || r.body.length === 0;
      }
      return false;
    }
  });
}

export function handleSummary(data) {
  return {
    "stdout": textSummary(data, { indent: " ", enableColors: true }),
    "../logs/k6-tiles-summary.json": JSON.stringify(data)
  };
}