import http from "k6/http";
import { sleep, check } from "k6";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.4/index.js";

export const options = {
  scenarios: {
    // Quick SQL validation - very light load
    baseline: { 
      executor: "shared-iterations", 
      vus: 1, 
      iterations: 3,   // Just 3 queries for quick test
      maxDuration: "60s" 
    }
  },
  thresholds: {
    http_req_failed: ["rate<0.2"],           // Allow 20% errors (very lenient for complex AI queries)
    http_req_duration: ['p(90)<20000', 'p(95)<30000'],      // 15 second timeout for AI queries
    http_req_duration: ["avg<8000"]          // 8 second average
  }
};

const BASE = __ENV.SQL_AGENT_BASE;

function pick(arr) { 
  return arr[Math.floor(Math.random() * arr.length)]; 
}

// Sample questions that the SQL agent should be able to handle
const questions = [
  // Basic property queries
  "Show me 5 properties under $500,000",
  "Find properties with 3 bedrooms",
  "What are the most expensive properties?",
  "Show me condos in Florida",
  "Find properties with more than 2000 square feet",
  
  // Location-based queries
  "What properties are available in Miami?",
  "Show me houses in California",
  "Find properties in New York state",
  "What are the cheapest properties in Texas?",
  
  // Aggregate queries
  "What is the average price by state?",
  "Show me the price range for 2 bedroom properties",
  "What states have the most expensive properties?",
  "Calculate average square footage by property type",
  
  // Complex queries
  "Find luxury properties with 4+ bedrooms over $1 million",
  "What is the price per square foot in different states?",
  "Show me properties sorted by price",
  "Find the best value properties (price per square foot)",
  
  // Market analysis
  "Which cities have the highest property values?",
  "Show me market trends by state",
  "What is the distribution of property types?",
  "Find undervalued properties in good locations"
];

export default function () {
  const question = pick(questions);
  
  const payload = {
    question: question
  };
  
  const url = `${BASE}/ask`;
  const res = http.post(url, JSON.stringify(payload), {
    headers: { 
      'Content-Type': 'application/json' 
    },
    tags: { 
      svc: "sql_agent",
      query_type: categorizeQuestion(question)
    },
    timeout: "10s"  // SQL queries can take time
  });
  
  check(res, {
    "success": (r) => r.status === 200,
    "has_answer": (r) => {
      try { 
        const data = r.json();
        return data.hasOwnProperty('answer') && data.answer && data.answer.length > 0;
      } catch { 
        return false; 
      }
    },
    "no_error_response": (r) => {
      try {
        const data = r.json();
        return !data.hasOwnProperty('error');
      } catch {
        return false;
      }
    },
    "reasonable_response_time": (r) => r.timings.duration < 10000  // 10 second max
  });
  
  // Longer sleep for SQL queries to avoid overwhelming the database
  sleep(2);
}

function categorizeQuestion(question) {
  const lower = question.toLowerCase();
  
  if (lower.includes('average') || lower.includes('calculate') || lower.includes('trends')) {
    return 'aggregate';
  } else if (lower.includes('find') || lower.includes('show') || lower.includes('properties')) {
    return 'search';
  } else if (lower.includes('under') || lower.includes('over') || lower.includes('price')) {
    return 'filter';
  } else if (lower.includes('state') || lower.includes('city') || lower.includes('location')) {
    return 'location';
  } else {
    return 'general';
  }
}

export function handleSummary(data) {
  return {
    "stdout": textSummary(data, { indent: " ", enableColors: true }),
    "../logs/k6-sql-agent-summary.json": JSON.stringify(data)
  };
}