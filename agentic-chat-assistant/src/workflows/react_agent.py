"""
LangGraph ReAct Agent Workflow - FIXED VERSION with Corrected Tool Execution
FIXES:
- Proper tool execution (invokes tools directly, not through ToolExecutor)
- Robust JSON parsing for intent classification
- Better error handling and logging
"""
import json
import re
from typing import Dict, List, Any, TypedDict, Annotated, Sequence
import operator
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import structlog

from src.config import settings
from src.tools.tools import TOOLS
from src.agents.supabase_domain import SUPABASE_DOMAIN_KNOWLEDGE
from src.agents.bigquery_domain import BIGQUERY_DOMAIN_KNOWLEDGE

logger = structlog.get_logger()


# ==================== STATE DEFINITION ====================

class AgentState(TypedDict):
    """State for the ReAct agent workflow"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    session_id: str
    intent: str
    requires_sql: bool
    requires_geocoding: bool
    requires_web_search: bool
    context: Dict[str, Any]
    iteration_count: int
    max_iterations: int


# ==================== INTENT CLASSIFICATION ====================

class IntentClassifier:
    """Classify user intent using Gemini 2.5 Flash with robust JSON parsing"""
    
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model=settings.gemini_flash_model,
            temperature=0.1,
            google_api_key=settings.google_api_key
        )
        
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for a real estate assistant. 
Analyze the user query and respond with ONLY valid JSON (no markdown, no extra text).

Output Format (JSON only):
{{
  "primary_intent": "INTENT_NAME",
  "secondary_intents": ["INTENT_NAME"],
  "requires_supabase_sql": true/false,
  "requires_bigquery_sql": true/false,
  "requires_geocoding": true/false,
  "requires_web_search": true/false,
  "reasoning": "brief explanation"
}}

Intents:
- PROPERTY_SEARCH: Find properties
- MARKET_ANALYSIS: Market statistics, trends, predictions
- INVESTMENT_ADVICE: Investment recommendations
- RISK_ASSESSMENT: Disaster risk or safety information
- LOCATION_BASED: Mentions specific locations needing coordinates
- AGENT_FINDER: Real estate agent recommendations
- WEB_SEARCH: Current events or information not in database
- GENERAL_CHAT: General questions about real estate

SQL Requirements:
- requires_supabase_sql: Property listings, risk data, ZIP codes
- requires_bigquery_sql: Market statistics, trends, predictions

Additional Flags:
- requires_geocoding: Universities, landmarks, addresses
- requires_web_search: Current news, recent events"""),
            ("human", "User Query: {query}\n\nRespond with JSON only:")
        ])
        
        self.parser = JsonOutputParser()
    
    def _extract_json(self, content: str) -> Dict[str, Any]:
        """Robustly extract JSON from response content"""
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass
        
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'`(.*?)`',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        return json.loads(match.strip())
                    except json.JSONDecodeError:
                        continue
        
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if "primary_intent" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        logger.warning("json_extraction_fallback", content=content[:200])
        
        intent_match = re.search(r'"primary_intent"\s*:\s*"([^"]+)"', content)
        if intent_match:
            return {
                "primary_intent": intent_match.group(1),
                "secondary_intents": [],
                "requires_supabase_sql": "property" in content.lower() or "risk" in content.lower(),
                "requires_bigquery_sql": "market" in content.lower() or "trend" in content.lower(),
                "requires_geocoding": "location" in content.lower() or "near" in content.lower(),
                "requires_web_search": "current" in content.lower() or "recent" in content.lower(),
                "reasoning": "Extracted from partial response"
            }
        
        raise ValueError(f"Could not extract valid JSON from response")
    
    def classify(self, query: str) -> Dict[str, Any]:
        """Classify user intent with robust error handling"""
        try:
            messages = self.intent_prompt.format_messages(query=query)
            response = self.model.invoke(messages)
            content = response.content
            
            logger.info("intent_classification_raw_response", 
                       query=query[:100], 
                       response_preview=content[:200])
            
            intent_data = self._extract_json(content)
            
            if "primary_intent" not in intent_data:
                raise ValueError("Missing primary_intent in response")
            
            defaults = {
                "secondary_intents": [],
                "requires_supabase_sql": False,
                "requires_bigquery_sql": False,
                "requires_geocoding": False,
                "requires_web_search": False,
                "reasoning": ""
            }
            
            for key, default_value in defaults.items():
                if key not in intent_data:
                    intent_data[key] = default_value
            
            logger.info("intent_classified_successfully", 
                       query=query[:100],
                       intent=intent_data["primary_intent"])
            
            return intent_data
            
        except Exception as e:
            logger.error("intent_classification_error", 
                        error=str(e),
                        error_type=type(e).__name__,
                        query=query[:100],
                        response_content=content[:500] if 'content' in locals() else "No response")
            
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['property', 'house', 'home', 'condo', 'apartment']):
                primary_intent = "PROPERTY_SEARCH"
                requires_supabase = True
                requires_bigquery = False
            elif any(word in query_lower for word in ['market', 'trend', 'price', 'statistics']):
                primary_intent = "MARKET_ANALYSIS"
                requires_supabase = False
                requires_bigquery = True
            elif any(word in query_lower for word in ['investment', 'invest', 'buy', 'opportunity']):
                primary_intent = "INVESTMENT_ADVICE"
                requires_supabase = True
                requires_bigquery = True
            elif any(word in query_lower for word in ['risk', 'safe', 'disaster', 'flood', 'earthquake']):
                primary_intent = "RISK_ASSESSMENT"
                requires_supabase = True
                requires_bigquery = False
            elif any(word in query_lower for word in ['agent', 'realtor', 'broker']):
                primary_intent = "AGENT_FINDER"
                requires_supabase = False
                requires_bigquery = False
            else:
                primary_intent = "GENERAL_CHAT"
                requires_supabase = False
                requires_bigquery = False
            
            requires_geocoding = any(word in query_lower for word in [
                'near', 'close to', 'around', 'university', 'downtown', 
                'within', 'miles', 'distance'
            ])
            
            return {
                "primary_intent": primary_intent,
                "secondary_intents": [],
                "requires_supabase_sql": requires_supabase,
                "requires_bigquery_sql": requires_bigquery,
                "requires_geocoding": requires_geocoding,
                "requires_web_search": False,
                "reasoning": "Fallback classification using keyword matching"
            }


# ==================== AGENT NODES ====================

class ReActAgent:
    """ReAct (Reasoning + Acting) Agent with LangGraph - FIXED TOOL EXECUTION"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        
        self.reasoning_model = ChatGoogleGenerativeAI(
            model=settings.gemini_flash_model,
            temperature=0.3,
            google_api_key=settings.google_api_key
        ).bind_tools(TOOLS)
        
        self.formatter_model = ChatGoogleGenerativeAI(
            model=settings.gemini_flash_model,
            temperature=0.5,
            google_api_key=settings.google_api_key
        )
        
        # Create tool lookup dictionary for direct invocation
        self.tools_by_name = {tool.name: tool for tool in TOOLS}
        
        # Build workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("classify_intent", self.classify_intent_node)
        workflow.add_node("reason", self.reason_node)
        workflow.add_node("act", self.act_node)
        workflow.add_node("format_response", self.format_response_node)
        
        workflow.set_entry_point("classify_intent")
        workflow.add_edge("classify_intent", "reason")
        workflow.add_conditional_edges(
            "reason",
            self.should_continue,
            {
                "continue": "act",
                "end": "format_response"
            }
        )
        workflow.add_edge("act", "reason")
        workflow.add_edge("format_response", END)
        
        return workflow
    
    def classify_intent_node(self, state: AgentState) -> AgentState:
        """Classify user intent"""
        query = state["user_query"]
        intent_data = self.intent_classifier.classify(query)
        
        state["intent"] = intent_data.get("primary_intent", "GENERAL_CHAT")
        state["requires_sql"] = (
            intent_data.get("requires_supabase_sql", False) or
            intent_data.get("requires_bigquery_sql", False)
        )
        state["requires_geocoding"] = intent_data.get("requires_geocoding", False)
        state["requires_web_search"] = intent_data.get("requires_web_search", False)
        state["context"]["intent_analysis"] = intent_data
        
        logger.info("intent_classified_in_node", 
                   intent=state["intent"],
                   requires_sql=state["requires_sql"],
                   requires_geocoding=state["requires_geocoding"])
        
        return state
    
    def reason_node(self, state: AgentState) -> AgentState:
        """Reasoning node - decide what action to take"""
        messages = state["messages"]
        
        system_message = self._build_system_message(state)
        messages_with_system = [system_message] + list(messages)
        
        try:
            response = self.reasoning_model.invoke(messages_with_system)
            
            state["messages"] = state["messages"] + [response]
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            
            logger.info("reasoning_step_completed", 
                       iteration=state["iteration_count"],
                       has_tool_calls=bool(response.tool_calls))
            
        except Exception as e:
            logger.error("reasoning_error", error=str(e))
            error_msg = AIMessage(content=f"I encountered an error: {str(e)}. Let me try a different approach.")
            state["messages"] = state["messages"] + [error_msg]
        
        return state
    
    def act_node(self, state: AgentState) -> AgentState:
        """Action node - execute tools DIRECTLY (FIXED)"""
        last_message = state["messages"][-1]
        
        tool_outputs = []
        for tool_call in last_message.tool_calls:
            try:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.info("executing_tool", 
                           tool=tool_name, 
                           args=tool_args,
                           iteration=state["iteration_count"])
                
                # FIXED: Invoke tool directly instead of through ToolExecutor
                if tool_name not in self.tools_by_name:
                    raise ValueError(f"Tool '{tool_name}' not found in available tools")
                
                tool = self.tools_by_name[tool_name]
                output = tool.invoke(tool_args)  # Direct invocation
                
                tool_outputs.append(
                    ToolMessage(
                        content=json.dumps(output, indent=2),
                        tool_call_id=tool_call["id"]
                    )
                )
                
                logger.info("tool_executed_successfully", 
                           tool=tool_name,
                           success=output.get("success", False) if isinstance(output, dict) else True,
                           result_preview=str(output)[:200])
                
            except Exception as e:
                logger.error("tool_execution_error", 
                            tool=tool_name if 'tool_name' in locals() else "unknown", 
                            error=str(e),
                            error_type=type(e).__name__)
                
                tool_outputs.append(
                    ToolMessage(
                        content=json.dumps({
                            "error": str(e),
                            "success": False,
                            "tool": tool_name if 'tool_name' in locals() else "unknown"
                        }),
                        tool_call_id=tool_call.get("id", "unknown")
                    )
                )
        
        state["messages"] = state["messages"] + tool_outputs
        return state
    
    def format_response_node(self, state: AgentState) -> AgentState:
        """Format final response for user"""
        messages = state["messages"]
        
        format_prompt = f"""You are a helpful real estate assistant. Format the conversation results into a clear, user-friendly response.

Guidelines:
1. Start with a concise summary answering the user's question
2. Present key information using clear formatting:
   - Use **bold** for important numbers and names
   - Use bullet points for lists
   - Include relevant details like prices, locations, dates
3. For properties: Include address, price, beds/baths, key features
4. For market data: Include trends, predictions, and context
5. For investment advice: Include risks, opportunities, and recommendations
6. Always be conversational and helpful
7. If recommending agents, direct to: {settings.agent_finder_url}
8. End with a helpful follow-up question if appropriate

Previous conversation:
{self._format_conversation_history(messages[-5:])}

Format this into a clear, engaging response for the user:"""
        
        try:
            formatted_response = self.formatter_model.invoke([HumanMessage(content=format_prompt)])
            
            state["messages"] = state["messages"] + [formatted_response]
            state["context"]["final_response"] = formatted_response.content
            
            logger.info("response_formatted_successfully")
            
        except Exception as e:
            logger.error("response_formatting_error", error=str(e))
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    state["context"]["final_response"] = msg.content
                    break
        
        return state
    
    def should_continue(self, state: AgentState) -> str:
        """Determine if we should continue or end"""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if state["iteration_count"] >= state["max_iterations"]:
                logger.warning("max_iterations_reached", count=state["iteration_count"])
                return "end"
            return "continue"
        
        return "end"
    
    def _build_system_message(self, state: AgentState) -> SystemMessage:
        """Build system message with context and domain knowledge"""
        intent = state.get("intent", "GENERAL_CHAT")
        intent_analysis = state["context"].get("intent_analysis", {})
        
        system_prompt = f"""You are an expert real estate assistant with access to comprehensive property data, market analytics, and risk assessment tools.

**Current Intent**: {intent}
**Analysis**: {intent_analysis.get('reasoning', 'N/A')}

**Your Capabilities:**
1. **Property Search**: Query Supabase for property listings using query_supabase_natural_language tool
2. **Market Analysis**: Query BigQuery for market trends using query_bigquery_natural_language tool
3. **Geocoding**: Convert locations to coordinates using geocode_location tool
4. **Distance Calculations**: Find properties near specific locations
5. **Risk Assessment**: Analyze disaster risk using FEMA NRI data
6. **Investment Analysis**: Combine risk, market, and prediction data
7. **Web Search**: Get current information using search_web tool
8. **Agent Finder**: Direct users to agent finder using find_real_estate_agents tool

**CRITICAL SQL RULES:**
- ONLY use SELECT queries - no INSERT, UPDATE, DELETE, DROP, etc.
- NEVER query these protected tables: profiles, user_favorites, users, sessions, auth
- Always check query syntax before execution
- Use proper table names and column names from domain knowledge
- Add LIMIT clauses to prevent large result sets

**For agent recommendations, ALWAYS use find_real_estate_agents tool**

**Domain Knowledge Summary:**

## SUPABASE TABLES (PostgreSQL + PostGIS)
Tables: properties, nri_counties, uszips, gis.us_counties
- properties: Active listings with lat/lon, price, beds/baths, risk data
- nri_counties: FEMA disaster risk by county (county_fips field)
- uszips: ZIP code database
- gis.us_counties: County polygons for spatial queries (geoid field)

FIPS Mapping: properties.county_geoid = nri_counties.county_fips = gis.us_counties.geoid

## BIGQUERY TABLES
Tables: county_market, county_predictions, state_market, state_predictions, county_lookup, state_lookup
Dataset: `{settings.bigquery_dataset_full}`
- Market statistics and predictions
- buyer_friendly field: 1 = buyer market, 0 = seller market
- market_trend field: "increasing", "stable", "declining"

**Query Examples:**

Supabase (Property Search using natural language):
Use query_supabase_natural_language tool with question like:
"Find 3-bedroom houses in Boston under $800k"

BigQuery (Market Analysis using natural language):
Use query_bigquery_natural_language tool with question like:
"What are the real estate trends in Texas for 2025?"

**Response Style:**
- Be conversational and helpful
- Present information clearly with formatting
- Include specific numbers, prices, and locations
- Provide actionable recommendations
- Offer to search more or adjust criteria

Now help the user with their query using the appropriate tools."""
        
        return SystemMessage(content=system_prompt)
    
    def _format_conversation_history(self, messages: List[BaseMessage]) -> str:
        """Format conversation history for display"""
        formatted = []
        for msg in messages:
            msg_type = msg.__class__.__name__
            content = str(msg.content)[:500]
            if len(str(msg.content)) > 500:
                content += "..."
            formatted.append(f"{msg_type}: {content}")
        return "\n".join(formatted)
    
    def run(self, query: str, session_id: str, history: List[Dict] = None) -> Dict[str, Any]:
        """Run the agent synchronously"""
        logger.info("agent_run_started", query=query[:100], session_id=session_id)
        
        messages = []
        if history:
            for msg in history[-10:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=query))
        
        initial_state = AgentState(
            messages=messages,
            user_query=query,
            session_id=session_id,
            intent="",
            requires_sql=False,
            requires_geocoding=False,
            requires_web_search=False,
            context={
                "start_time": datetime.now().isoformat()
            },
            iteration_count=0,
            max_iterations=10
        )
        
        try:
            final_state = self.app.invoke(initial_state)
            
            final_response = final_state["context"].get("final_response", "")
            if not final_response:
                for msg in reversed(final_state["messages"]):
                    if isinstance(msg, AIMessage):
                        final_response = msg.content
                        break
            
            if not final_response:
                final_response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            logger.info("agent_run_completed", 
                       iterations=final_state["iteration_count"],
                       intent=final_state["intent"],
                       success=True)
            
            return {
                "success": True,
                "response": final_response,
                "intent": final_state["intent"],
                "iterations": final_state["iteration_count"],
                "context": final_state["context"]
            }
            
        except Exception as e:
            logger.error("agent_run_error", 
                        error=str(e), 
                        error_type=type(e).__name__,
                        query=query[:100])
            
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error processing your request. Please try rephrasing your question or contact support if the issue persists."
            }


# Global agent instance
agent = ReActAgent()