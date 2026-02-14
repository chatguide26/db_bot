"""
Enhanced Streamlit App with Agno Agent + Knowledge Base + Teradata Integration
"""
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import logging
import requests
import hashlib
import math
from typing import Optional
from datetime import datetime

# Local imports
from src.connectors import TeradataConnector, DuckDBConnector, SmartDataProcessor
from src.knowledge_manager import KnowledgeManager
from src.agent_tools import create_tool_functions
from src.visualizations import ChartBuilder, ChartType, VisualizationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CACHING SYSTEM ====================
class ResponseCache:
    """Simple in-memory cache for LLM responses"""
    def __init__(self, max_size=100, ttl_seconds=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _hash_key(self, text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, key: str):
        h = self._hash_key(key)
        if h in self.cache:
            entry = self.cache[h]
            age = time.time() - entry['timestamp']
            if age < self.ttl_seconds:
                logger.info(f"✅ Cache HIT for: {key[:50]}...")
                return entry['value']
            else:
                del self.cache[h]
        return None
    
    def set(self, key: str, value):
        h = self._hash_key(key)
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest = min(self.cache.items(), key=lambda x: x[1]['timestamp'])
            del self.cache[oldest[0]]
        self.cache[h] = {
            'value': value,
            'timestamp': time.time()
        }
        logger.info(f"💾 Cached: {key[:50]}... (Size: {len(self.cache)})")
    
    def clear(self):
        self.cache.clear()

# Initialize cache
response_cache = ResponseCache(max_size=50, ttl_seconds=3600)

# ==================== DATABASE SCHEMA CACHE ====================
schema_cache = {}

def get_database_schema(duckdb_conn) -> dict:
    """Cache and return database schema (table names and columns)"""
    global schema_cache
    
    if "schema" in schema_cache:
        logger.info("✅ Using cached database schema")
        return schema_cache["schema"]
    
    schema = {}
    try:
        tables = duckdb_conn.list_tables()
        for table_name in tables:
            try:
                # Get columns for this table
                result = duckdb_conn.query(f"SELECT * FROM {table_name} LIMIT 0")
                schema[table_name] = list(result.columns)
            except Exception as e:
                logger.warning(f"Could not get columns for {table_name}: {str(e)}")
                schema[table_name] = []
        
        schema_cache["schema"] = schema
        logger.info(f"💾 Cached schema: {len(schema)} tables")
        return schema
    except Exception as e:
        logger.error(f"Error getting schema: {str(e)}")
        return {}

def format_schema_response(schema: dict) -> str:
    """Format schema as readable markdown table"""
    md = "# 📊 Database Schema\n\n"
    for table_name, columns in schema.items():
        md += f"## {table_name}\n"
        md += "| Column Name |\n"
        md += "|---|\n"
        for col in columns:
            md += f"| {col} |\n"
        md += "\n"
    return md

# ==================== SPECIAL HANDLERS ====================

def is_greeting(text: str) -> bool:
    """Detect if input is a greeting (no LLM needed)"""
    greetings = {
        'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
        'howdy', 'hiya', 'yo', 'what up', "what's up", 'whatsup', 'how are you',
        'how r u', "how're you", 'thanks', 'thank you', 'appreciate it', 'thx'
    }
    text_lower = text.lower().strip()
    
    # Check exact matches or contains
    return text_lower in greetings or any(g in text_lower for g in greetings if len(g) > 3)

def handle_greeting(text: str) -> str:
    """Handle greeting without calling LLM"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['thank', 'thx', 'appreciate']):
        return "You're welcome! 😊 Is there anything else you'd like to know about the subscriber data?"
    
    return """Hello! 👋 I'm your Fixed Broadband Intelligence Assistant. 

I can help you with:
- **ARPU Analysis**: Average revenue per user trends and segmentation
- **Churn Risk**: Identify at-risk subscribers
- **Technology Mix**: Current technology adoption and migration opportunities  
- **Demographics**: Subscriber segmentation by age, location, etc.
- **Database Schema**: Show available tables and columns

What would you like to analyze?"""

def normalize_question(text: str) -> str:
    """Normalize question for better cache matching"""
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Normalize common variations
    text = text.replace('whats', "what's")
    text = text.replace('hows', "how's")
    text = text.replace('wheres', "where's")
    text = text.replace('show me', '')
    text = text.replace('tell me', '')
    text = text.replace('can you', '')
    text = text.replace('could you', '')
    text = text.replace('give me', '')
    text = text.replace('please', '')
    
    # Remove punctuation
    text = text.replace('?', '')
    text = text.replace('!', '')
    
    # Clean up extra spaces again
    text = ' '.join(text.split())
    
    return text

# ==================== FAQ GENERATION ====================
def generate_recommended_faqs(duckdb_conn, km: KnowledgeManager) -> list:
    """
    Analyze loaded data and generate FAQ questions that are GUARANTEED to work
    Returns list of (question, sql_query) tuples
    """
    try:
        # Get data statistics
        stats = duckdb_conn.query("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT Current_Package_Name) as package_count,
                COUNT(DISTINCT Payment_Type) as payment_type_count,
                COUNT(DISTINCT GOV) as gov_count,
                COUNT(DISTINCT Has_GSM) as gsm_type_count
            FROM broadband_subscribers
        """).iloc[0].to_dict()
        
        faqs = []
        
        # FAQ 1: Service Quality by Package
        faqs.append({
            "question": "What is the service quality (outages and tickets) by package type?",
            "sql": """
                SELECT Current_Package_Name as package,
                       COUNT(*) as subscriber_count,
                       ROUND(AVG(Count_Of_All_Outage), 2) as avg_outages,
                       ROUND(AVG(TTS_Total_Tickets), 2) as avg_tickets,
                       ROUND(MAX(Count_Of_All_Outage), 0) as max_outages
                FROM broadband_subscribers
                WHERE Current_Package_Name IS NOT NULL
                GROUP BY Current_Package_Name
                ORDER BY avg_outages DESC
            """,
            "description": "Compare service outages and support tickets across different broadband packages"
        })
        
        # FAQ 2: Total Revenue by Payment Type
        faqs.append({
            "question": "What is the total revenue generated by payment type (Prepaid vs Postpaid)?",
            "sql": """
                SELECT Payment_Type,
                       COUNT(*) as subscriber_count,
                       ROUND(SUM(Total_RPU), 0) as total_revenue_egp,
                       ROUND(AVG(Total_RPU), 0) as avg_revenue_egp,
                       ROUND(AVG(Tenure_Days), 0) as avg_tenure_days
                FROM broadband_subscribers
                WHERE Payment_Type IS NOT NULL
                GROUP BY Payment_Type
                ORDER BY total_revenue_egp DESC
            """,
            "description": "Compare revenue generation between Prepaid and Postpaid customers"
        })
        
        # FAQ 3: Subscriber distribution by Status
        faqs.append({
            "question": "How many subscribers are Active, Suspended, or Deactivated?",
            "sql": """
                SELECT Subscriber_Status,
                       COUNT(*) as subscriber_count,
                       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM broadband_subscribers), 1) as percentage,
                       ROUND(AVG(Total_RPU), 0) as avg_revenue_egp
                FROM broadband_subscribers
                WHERE Subscriber_Status IS NOT NULL
                GROUP BY Subscriber_Status
                ORDER BY subscriber_count DESC
            """,
            "description": "View subscriber distribution by operational status"
        })
        
        # FAQ 4: Service Reliability Analysis
        faqs.append({
            "question": "Which packages have the highest service reliability (fewest outages)?",
            "sql": """
                SELECT Current_Package_Name as package,
                       COUNT(*) as subscriber_count,
                       ROUND(AVG(Count_Of_All_Outage), 2) as avg_outages,
                       ROUND(AVG(Total_RPU), 0) as avg_revenue_egp
                FROM broadband_subscribers
                WHERE Current_Package_Name IS NOT NULL AND Count_Of_All_Outage IS NOT NULL
                GROUP BY Current_Package_Name
                ORDER BY avg_outages ASC
            """,
            "description": "Identify packages with best service reliability and outage records"
        })
        
        # FAQ 5: High-Value Customer Analysis
        faqs.append({
            "question": "Who are the highest-value subscribers (highest Total Revenue)?",
            "sql": """
                SELECT Subs_ID, Service_Number_Trimmed, Current_Package_Name, 
                       Payment_Type, Total_RPU, Tenure_Days,
                       Count_Of_All_Outage, TTS_Total_Tickets
                FROM broadband_subscribers
                WHERE Total_RPU IS NOT NULL
                ORDER BY Total_RPU DESC
                LIMIT 15
            """,
            "description": "View top subscribers by lifetime revenue for VIP programs"
        })
        
        # FAQ 6: Cross-sell Opportunity Analysis
        faqs.append({
            "question": "How many subscribers have GSM service? What is their revenue potential?",
            "sql": """
                SELECT Has_GSM,
                       COUNT(*) as subscriber_count,
                       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM broadband_subscribers), 1) as percentage,
                       ROUND(AVG(Total_RPU), 0) as avg_revenue_egp,
                       ROUND(AVG(Tenure_Days), 0) as avg_tenure_days
                FROM broadband_subscribers
                WHERE Has_GSM IS NOT NULL
                GROUP BY Has_GSM
            """,
            "description": "Identify GSM cross-sell and bundling opportunities"
        })
        
        # FAQ 7: Package Upgrade Opportunity
        faqs.append({
            "question": "Which subscribers have low support tickets and could be upsold to premium packages?",
            "sql": """
                SELECT Current_Package_Name as current_package,
                       COUNT(*) as subscriber_count,
                       ROUND(AVG(TTS_Total_Tickets), 2) as avg_tickets,
                       ROUND(AVG(Total_RPU), 0) as avg_revenue_egp,
                       ROUND(AVG(Tenure_Days), 0) as avg_tenure_days
                FROM broadband_subscribers
                WHERE Current_Package_Name IS NOT NULL
                  AND TTS_Total_Tickets < 3
                GROUP BY Current_Package_Name
                ORDER BY avg_tenure_days DESC
            """,
            "description": "Identify satisfied, low-complaint customers ready for premium upgrades"
        })
        
        # FAQ 8: Problem Subscribers Analysis
        faqs.append({
            "question": "Which subscribers have the highest support tickets or outages?",
            "sql": """
                SELECT Subs_ID, Service_Number_Trimmed, Current_Package_Name,
                       TTS_Total_Tickets, Count_Of_All_Outage, Total_RPU
                FROM broadband_subscribers
                WHERE TTS_Total_Tickets IS NOT NULL OR Count_Of_All_Outage IS NOT NULL
                ORDER BY (TTS_Total_Tickets + Count_Of_All_Outage) DESC
                LIMIT 15
            """,
            "description": "Identify problem subscribers needing technical support or investigation"
        })
        
        # FAQ 9: Geographic Performance Analysis
        faqs.append({
            "question": "What is the revenue and service quality by governorate?",
            "sql": """
                SELECT GOV as governorate,
                       COUNT(*) as subscriber_count,
                       ROUND(AVG(Total_RPU), 0) as avg_revenue_egp,
                       ROUND(SUM(Total_RPU), 0) as total_revenue_egp,
                       ROUND(AVG(Count_Of_All_Outage), 2) as avg_outages
                FROM broadband_subscribers
                WHERE GOV IS NOT NULL
                GROUP BY GOV
                ORDER BY avg_revenue_egp DESC
            """,
            "description": "Compare revenue and service quality across Egyptian governorates"
        })
        
        # FAQ 10: Gender Demographics and Revenue
        faqs.append({
            "question": "What is the subscriber distribution and revenue by gender?",
            "sql": """
                SELECT Gender,
                       COUNT(*) as subscriber_count,
                       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM broadband_subscribers), 1) as percentage,
                       ROUND(AVG(Total_RPU), 0) as avg_revenue_egp,
                       ROUND(AVG(Tenure_Days), 0) as avg_tenure_days
                FROM broadband_subscribers
                WHERE Gender IS NOT NULL
                GROUP BY Gender
            """,
            "description": "Analyze subscriber base and revenue by gender demographics"
        })
        
        logger.info(f"✅ Generated {len(faqs)} recommended FAQs")
        return faqs
        
    except Exception as e:
        logger.error(f"Error generating FAQs: {str(e)}")
        return []

# ==================== CONFIGURATION ====================
TERADATA_CONFIG = {
    "host": os.getenv("TERADATA_HOST", "10.19.199.28"),
    "username": os.getenv("TERADATA_USER", "mostafa_farouk"),
    "password": os.getenv("TERADATA_PASSWORD", "Qx7$LMNOPQRN"),
    "database": os.getenv("TERADATA_DB", "Tedata_temp")
}
LLM_CONFIG = {
    "base_url": os.getenv("LLM_BASE_URL", "http://192.168.120.227:7070"),
    "model": os.getenv("LLM_MODEL", "agpt-oss-20b")
}
BASE_QUERY = """
SELECT Subs_ID, Service_Number_Trimmed, Subscriber_Status,
       Payment_Type, TTS_Total_Tickets, Count_Of_All_Outage,
       Current_Package_Name, Total_RPU, Tenure_Days,
       Gender, Has_GSM, GOV
FROM fixed_broadband.subscriber_profile
WHERE GOV = '{gov}'
  AND Subscriber_Status IS NOT NULL
"""

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🧠 Fixed Broadband Analytics Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Fixed Broadband Analytics Intelligence Agent")
st.markdown("**Teradata → DuckDB → Intelligent Agent = Network Performance & Revenue Insights**")

# ==================== SESSION STATE ====================
if "connection" not in st.session_state:
    st.session_state.connection = None
if "duckdb" not in st.session_state:
    st.session_state.duckdb = None
if "data" not in st.session_state:
    st.session_state.data = None
if "knowledge_manager" not in st.session_state:
    st.session_state.knowledge_manager = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "tools" not in st.session_state:
    st.session_state.tools = None
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None
if "recommended_faqs" not in st.session_state:
    st.session_state.recommended_faqs = []
if "user_selected_faq" not in st.session_state:
    st.session_state.user_selected_faq = None

# ==================== FUNCTIONS ====================
def detect_llm_model():
    """Detect available LLM model from LM Studio"""
    try:
        base_url = LLM_CONFIG['base_url'].rstrip('/')
        endpoints = [
            f"{base_url}/v1/models",
            f"{base_url}/api/models",
        ]
        
        for endpoint in endpoints:
            try:
                logger.info(f"🔍 Trying model detection at: {endpoint}")
                response = requests.get(endpoint, timeout=10, proxies={"http": None, "https": None})
                if response.status_code == 200:
                    data = response.json()
                    models = data.get('data', [])
                    if models:
                        model_name = models[0].get('id', 'agpt-oss-20b')
                        logger.info(f"✅ Detected model: {model_name}")
                        return model_name
            except Exception as e:
                logger.debug(f"Model detection failed at {endpoint}: {str(e)}")
                continue
    except Exception as e:
        logger.warning(f"Could not detect LLM model: {str(e)}")
    
    return "agpt-oss-20b"

# ==================== EMBEDDING & SIMILARITY FUNCTIONS ====================
def get_embedding(text: str, model: str = "text-embedding-all-minilm-l12-v2") -> Optional[list]:
    """
    Get embedding vector from LM Studio embeddings API
    Uses: all-MiniLM-L12-v2.F16 (recommended for better quality)
    """
    try:
        response = requests.post(
            f"{LLM_CONFIG['base_url']}/v1/embeddings",
            json={
                "model": model,
                "input": text
            },
            timeout=30,
            proxies={"http": None, "https": None}
        )
        
        if response.status_code == 200:
            data = response.json()
            embedding = data.get('data', [{}])[0].get('embedding', None)
            logger.debug(f"✅ Got embedding for: {text[:40]}...")
            return embedding
        else:
            logger.error(f"Embedding API error: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        return None


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calculate cosine similarity between two vectors
    """
    if not vec1 or not vec2:
        return 0.0
    
    # Dot product
    dot = sum(a * b for a, b in zip(vec1, vec2))
    
    # Magnitudes
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot / (mag1 * mag2)


# ==================== SEMANTIC SCHEMA MATCHING ====================
embedding_cache = {}

def build_column_embeddings(duckdb_conn) -> dict:
    """
    Build semantic embeddings for all database columns at initialization.
    Caches embeddings to avoid repeated API calls.
    
    Returns: {column_name: embedding_vector}
    """
    global embedding_cache
    
    # Check if already cached
    if "columns" in embedding_cache:
        logger.info(f"✅ Using cached column embeddings ({len(embedding_cache['columns'])} columns)")
        return embedding_cache["columns"]
    
    # Column descriptions for better semantic understanding
    column_descriptions = {
        "Subs_ID": "Unique internal identifier for each fixed broadband subscriber record",
        "Service_Number_Trimmed": "Subscriber service or line number used for identification and billing",
        "Subscriber_Status": "Current operational status (Active, Suspend, Deactivation) used to identify active and inactive customers",
        "Payment_Type": "Billing type (Prepaid or Postpaid)",
        "TTS_Total_Tickets": "Total number of technical support or trouble tickets raised by the subscriber",
        "Count_Of_All_Outage": "Total number of service outage incidents experienced by the subscriber",
        "Current_Package_Name": "Name of the current broadband package or tariff plan subscribed to",
        "Total_RPU": "Total lifetime revenue generated by the subscriber in Egyptian pounds",
        "Tenure_Days": "Number of days since subscriber activation (customer lifetime)",
        "Gender": "Subscriber gender (Male or Female)",
        "Has_GSM": "Indicates whether the subscriber also has GSM mobile service (Yes or No)",
        "GOV": "Governorate or geographic region where the subscriber is located"
    }
    
    column_embeddings = {}
    
    logger.info(f"🔄 Building embeddings for {len(column_descriptions)} columns...")
    
    for col, description in column_descriptions.items():
        # Use description for richer semantic context
        embedding = get_embedding(description)
        if embedding:
            column_embeddings[col] = embedding
            logger.debug(f"  ✅ {col}")
        else:
            logger.warning(f"  ⚠️ Failed to embed column: {col}")
    
    # Cache the embeddings
    embedding_cache["columns"] = column_embeddings
    logger.info(f"✅ Column embeddings cached: {len(column_embeddings)} columns")
    
    return column_embeddings


def find_relevant_columns(user_question: str, column_embeddings: dict, top_k: int = 3) -> list:
    """
    Find the most relevant columns for a vague question using semantic similarity.
    
    Args:
        user_question: The user's question
        column_embeddings: Dict of {column_name: embedding_vector}
        top_k: Number of top columns to return (default: 3)
    
    Returns:
        List of (column_name, similarity_score) tuples, sorted by relevance
    """
    # Get embedding for the user's question
    question_embedding = get_embedding(user_question)
    
    if not question_embedding:
        logger.warning(f"Could not embed question: {user_question[:40]}...")
        return []
    
    # Calculate similarity with each column
    similarities = []
    for col_name, col_embedding in column_embeddings.items():
        similarity = cosine_similarity(question_embedding, col_embedding)
        similarities.append((col_name, similarity))
    
    # Sort by similarity (descending) and get top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_columns = similarities[:top_k]
    
    logger.info(f"🎯 Relevant columns for '{user_question[:40]}...':")
    for col, score in top_columns:
        logger.info(f"  {col}: {score:.2f}")
    
    return top_columns

# ==================== SIDEBAR CONTROLS ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM Status
    st.markdown("### 🤖 LLM Status")
    if st.session_state.llm_model:
        st.success(f"✅ Running: **{st.session_state.llm_model}**")
        st.caption(f"Endpoint: {LLM_CONFIG['base_url']}")
    else:
        with st.spinner("🔍 Detecting LLM..."):
            st.session_state.llm_model = detect_llm_model()
            if st.session_state.llm_model:
                st.success(f"✅ Detected: **{st.session_state.llm_model}**")
            else:
                st.warning("⚠️ LLM not detected")
                st.caption(f"Trying: {LLM_CONFIG['base_url']}")
    
    st.markdown("---")
    
    # Governorate selection
    gov = st.selectbox("📍 Select Governorate", ["Giza", "Cairo", "Alexandria", "Qalyubia"])
    
    # Connection button
    if st.button("🚀 Initialize Agent", type="primary"):
        with st.spinner("🔄 Connecting to Teradata..."):
            try:
                # Connect to Teradata
                teradata = TeradataConnector(**TERADATA_CONFIG)
                if teradata.connect():
                    # Query data
                    query = BASE_QUERY.format(gov=gov)
                    df = teradata.query(query)
                    teradata.close()
                    
                    if df is not None and len(df) > 0:
                        # Preprocess data
                        df = SmartDataProcessor.preprocess_subscriber_data(df)
                        
                        # Load to DuckDB
                        duckdb_conn = DuckDBConnector()
                        duckdb_conn.load_dataframe(df, "broadband_subscribers")
                        
                        # Load knowledge base
                        km = KnowledgeManager("knowledge")
                        
                        # Generate recommended FAQs based on loaded data
                        faqs = generate_recommended_faqs(duckdb_conn, km)
                        
                        # Pre-build column embeddings for vague question handling
                        with st.spinner("🏗️ Building semantic schema embeddings..."):
                            build_column_embeddings(duckdb_conn)
                            st.caption("✅ Semantic embeddings ready (all-MiniLM-L12-v2)")
                        
                        # Save to session
                        st.session_state.data = df
                        st.session_state.duckdb = duckdb_conn
                        st.session_state.knowledge_manager = km
                        st.session_state.tools = create_tool_functions(duckdb_conn, km)
                        st.session_state.recommended_faqs = faqs
                        
                        st.success(f"✅ Agent initialized with {len(df):,} {gov} subscribers!")
                        st.info(f"📋 Generated {len(faqs)} FAQ questions + semantic schema matching for vague questions")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ No data returned from Teradata")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # Info section
    st.subheader("📚 Knowledge Base")
    if st.session_state.knowledge_manager:
        km = st.session_state.knowledge_manager
        st.write(f"📊 Tables: {len(km.tables)}")
        st.write(f"📋 Metrics: {len(km.business_rules)}")
        st.write(f"🔍 Patterns: {len(km.query_patterns)}")
        st.write(f"💡 Learnings: {len(km.learnings)}")
    
    st.markdown("---")
    
    # Clear button
    if st.button("🗑️ Clear Session"):
        if st.session_state.duckdb:
            st.session_state.duckdb.close()
        st.session_state = {}
        st.rerun()

# ==================== HELPER FUNCTIONS ====================

def call_llm(input_text: str, system_prompt: str = "", model: str = None, show_streaming: bool = False, use_cache: bool = True, db_context: str = "") -> dict:
    """
    Call LM Studio LLM using native API format with streaming support and caching
    Returns: {"answer": "...", "reasoning": "...", "tokens": {...}, "time": {...}, "cached": bool}
    """
    try:
        # Check cache first
        cache_key = f"{system_prompt}||{input_text}"
        if use_cache:
            cached_result = response_cache.get(cache_key)
            if cached_result:
                cached_result["cached"] = True
                return cached_result
        
        start_time = time.time()
        
        if not model:
            model = st.session_state.llm_model or "agpt-oss-20b"
        
        if not system_prompt:
            system_prompt = "You are a helpful fixed broadband analytics expert. Be concise and specific with numbers. Use EGP for currency."
        
        # Add database context to system prompt if provided
        if db_context:
            system_prompt = f"{system_prompt}\n\n{db_context}"
        
        headers = {"Content-Type": "application/json"}
        
        # LM Studio native format (verified working)
        payload = {
            "model": model,
            "system_prompt": system_prompt,
            "input": input_text
        }
        
        # Build endpoint
        base_url = LLM_CONFIG['base_url'].rstrip('/')
        url = f"{base_url}/api/v1/chat"
        
        logger.info(f"🤖 LLM Call: {input_text[:60]}...")
        
        # Streaming placeholder
        stream_placeholder = None
        if show_streaming:
            stream_placeholder = st.empty()
            stream_placeholder.info("🧠 Model is thinking...", icon="⏳")
        
        # CRITICAL: Disable proxy to avoid timeout
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=500,  # Increased from 60 to 120 seconds
            proxies={"http": None, "https": None}
        )
        
        elapsed = time.time() - start_time
        
        logger.info(f"   Status: {response.status_code} | Time: {elapsed:.2f}s")
        
        result_data = {
            "answer": None,
            "reasoning": None,
            "tokens": {"input": 0, "output": 0, "total": 0},
            "time": {"elapsed": elapsed},
            "cached": False
        }
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract tokens if available
            if 'usage' in result:
                result_data["tokens"] = {
                    "input": result['usage'].get('prompt_tokens', 0),
                    "output": result['usage'].get('completion_tokens', 0),
                    "total": result['usage'].get('total_tokens', 0)
                }
            
            # Parse response
            if 'output' in result:
                output = result['output']
                logger.info(f"   Output type: {type(output).__name__}")
                
                # Handle list of output objects with type and content
                reasoning_text = ""
                answer_text = ""
                
                if isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict):
                            item_type = item.get('type', '')
                            content = item.get('content', '')
                            
                            if item_type == 'reasoning':
                                reasoning_text += content
                            elif item_type == 'answer' or item_type == 'response':
                                answer_text += content
                            else:
                                # If no type, treat as answer
                                answer_text += content
                        else:
                            answer_text += str(item)
                    
                    # Update streaming if enabled
                    if show_streaming and stream_placeholder:
                        with stream_placeholder.container():
                            if reasoning_text:
                                with st.expander("🧠 **Model Reasoning** (click to expand)", expanded=False):
                                    st.markdown(reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text)
                            
                            if answer_text:
                                st.success("✅ Answer Generated")
                                st.write(answer_text)
                    
                    result_data["reasoning"] = reasoning_text if reasoning_text else None
                    result_data["answer"] = answer_text if answer_text else None
                
                # Handle string output
                elif isinstance(output, str):
                    result_data["answer"] = output
                    if show_streaming and stream_placeholder:
                        stream_placeholder.success(f"✅ {output[:100]}...")
                
                logger.info(f"✅ LLM Response: {len(str(result_data['answer']))} chars")
                
                # Cache the result
                if use_cache and (result_data["answer"] or result_data["reasoning"]):
                    response_cache.set(cache_key, result_data.copy())
                
                return result_data
            
            else:
                logger.warning(f"⚠️ No 'output' in response. Keys: {list(result.keys())}")
                return result_data
        
        else:
            error_text = response.text[:400]
            logger.error(f"❌ HTTP {response.status_code}: {error_text}")
            if show_streaming and stream_placeholder:
                stream_placeholder.error(f"❌ Error {response.status_code}: {error_text[:100]}")
            return result_data
                
    except requests.exceptions.Timeout:
        logger.error(f"❌ LLM Timeout (500 exceeded)")
        if show_streaming and stream_placeholder:
            stream_placeholder.error("❌ LLM Request timeout (500)")
        return {"answer": None, "reasoning": None, "tokens": {}, "time": {"elapsed": time.time() - start_time}, "cached": False}
    except Exception as e:
        logger.error(f"❌ LLM Error: {str(e)}")
        if show_streaming and stream_placeholder:
            stream_placeholder.error(f"❌ Error: {str(e)[:100]}")
        return {"answer": None, "reasoning": None, "tokens": {}, "time": {"elapsed": time.time() - start_time}, "cached": False}


def query_data_with_llm(user_question: str, duckdb_conn) -> tuple:
    """
    Use LLM to understand user question and query database
    Returns: (answer, data, charts)
    """
    
    # Build database context from actual schema
    try:
        db_stats = duckdb_conn.query("""
            SELECT 
                COUNT(*) as total_subscribers,
                COUNT(DISTINCT Current_Package_Name) as package_types,
                COUNT(DISTINCT Payment_Type) as payment_types,
                COUNT(DISTINCT GOV) as governorates,
                ROUND(AVG(Total_RPU), 2) as avg_revenue,
                ROUND(MAX(Total_RPU), 2) as max_revenue,
                ROUND(MIN(Total_RPU), 2) as min_revenue
            FROM broadband_subscribers
        """).df()
        
        stats_text = db_stats.iloc[0].to_dict()
        db_context = f"""🔒 CRITICAL CONSTRAINT - You MUST use ONLY this database:

**Table:** broadband_subscribers (Fixed Broadband subscriber profiles)
**Sample Columns:** Subs_ID, Service_Number_Trimmed, Current_Package_Name, Payment_Type, Total_RPU, Tenure_Days

**Dataset Stats:**
- Total: {int(stats_text['total_subscribers'])} fixed broadband subscribers
- Packages: {int(stats_text['package_types'])} types
- Governorates: {int(stats_text['governorates'])}
- Revenue: {stats_text['avg_revenue']} EGP avg ({stats_text['min_revenue']}-{stats_text['max_revenue']} range)

**You MUST:**
1. ONLY reference the {int(stats_text['total_subscribers'])} fixed broadband subscribers in this dataset
2. NOT provide external/global/industry data
3. NOT hallucinate data
4. Answer based on actual database queries only"""
    except Exception as e:
        logger.warning(f"Could not generate DB stats: {str(e)}")
        db_context = ""
    
    # STEP 1: Let LLM construct the SQL query (FORCED query planning)
    query_planning_prompt = f"""User Question: "{user_question}"

You MUST respond with the EXACT SQL query to answer this question using the broadband_subscribers table.

**CRITICAL RULES:**
1. For string comparisons (Gender, Current_Technology, etc), use UPPER() for case-insensitive matching
   Example: WHERE UPPER(Gender) = 'FEMALE' instead of WHERE Gender = 'Female'
2. Use COUNT(*) for counting records
3. Always include meaningful aliases for results

Return ONLY the SQL query in this format:
```sql
SELECT ... FROM giza_data ...
```

Do NOT provide any other text. Just the SQL query."""
    
    query_planning = call_llm(query_planning_prompt, 
        system_prompt="You are a SQL expert. Write ONLY the SQL query, nothing else.",
        db_context=db_context,
        show_streaming=False)
    
    # Check both answer and reasoning fields (LLM may put SQL in reasoning)
    query_text = (query_planning.get("answer") or query_planning.get("reasoning") or "").strip() if isinstance(query_planning, dict) else ""
    
    # Extract SQL from markdown code block if present
    if "```sql" in query_text:
        query_text = query_text.split("```sql")[1].split("```")[0].strip()
    elif "```" in query_text:
        query_text = query_text.split("```")[1].split("```")[0].strip()
    
    # STEP 2: Execute the SQL query
    data_summary = ""
    query_executed = False
    
    if query_text and len(query_text) > 10:
        try:
            logger.info(f"📊 Executing query:\n{query_text}")
            result = duckdb_conn.query(query_text)
            data_summary = result.to_string()
            query_executed = True
            logger.info(f"✅ Query executed successfully. Rows: {len(result)}")
        except Exception as e:
            logger.error(f"❌ Query execution failed: {str(e)}")
            query_executed = False
    
    # STEP 3: Generate final answer based on query results
    if data_summary and len(data_summary) > 10:
        # Show query and data in streaming
        final_prompt = f"""📊 **Query Executed:**
```sql
{query_text}
```

**Results:**
```
{data_summary}
```

User original question: "{user_question}"

Based on the query results above, provide a clear, specific answer.
Include key metrics and numbers from the data.
Be concise (2-3 sentences)."""
        
        final_response = call_llm(final_prompt,
            system_prompt="You are a fixed broadband analyst. Answer ONLY based on the query results shown. Do NOT add external data or generic advice.",
            show_streaming=True,
            use_cache=True,
            db_context="")  # Don't duplicate context here
        
        answer = final_response.get("answer", "")
        reasoning = final_response.get("reasoning", "")
        
        # Format response with query visibility
        response_text = f"""🔍 **Query Executed:**
```sql
{query_text}
```

📊 **Data Retrieved:**
```
{data_summary}
```

✅ **Analysis:**
{answer}"""
        
        return response_text, []
    
    else:
        # No data found from query
        fallback_prompt = f"""User Question: "{user_question}"

The SQL query returned no results. What should I tell the user?
Be helpful - explain what data is available or suggest a different angle."""
        
        fallback = call_llm(fallback_prompt,
            system_prompt="You are a helpful fixed broadband analyst. Acknowledge data limitations.",
            show_streaming=True,
            use_cache=True,
            db_context=db_context)
        
        return fallback.get("answer", "No results found"), []


def _format_table(data: list) -> str:
    """Format list of dicts as markdown table"""
    if not data or not isinstance(data, list) or len(data) == 0:
        return "No data available"
    
    # Convert to DataFrame for formatting
    try:
        df = pd.DataFrame(data)
        # Format currency columns
        for col in df.columns:
            if 'arpu' in col.lower() or 'revenue' in col.lower():
                df[col] = df[col].apply(lambda x: f"{x:.0f} EGP" if isinstance(x, (int, float)) else x)
        return df.to_markdown(index=False)
    except Exception as e:
        # Fallback to simple string representation
        return str(data)[:500]



def find_matching_faq(user_question: str, faqs: list) -> dict:
    """
    Use simple keyword matching to find if user question matches a FAQ
    Returns the matching FAQ or None
    """
    question_lower = user_question.lower()
    
    for faq in faqs:
        faq_question_lower = faq["question"].lower()
        
        # Simple keyword matching
        keywords_in_faq = set(faq_question_lower.split())
        keywords_in_user = set(question_lower.split())
        
        # If 70%+ keywords match, it's a probable match
        common_keywords = keywords_in_faq & keywords_in_user
        if len(common_keywords) >= max(len(keywords_in_faq) * 0.5, 3):
            return faq
    
    return None


def _route_question(prompt: str, tools: dict, km: KnowledgeManager, duckdb_conn, faqs: list = None) -> tuple:
    """
    LLM-First Router with FAQ Matching:
    0. Check for greeting -> instant response
    1. Check for schema questions -> cached schema
    2. Check if question matches a recommended FAQ -> execute pre-validated SQL
    3. If no FAQ match, use SQL-first approach with LLM
    Shows: reasoning -> answer + metrics
    """
    
    if not duckdb_conn:
        return "❌ Database not connected.", []
    
    if faqs is None:
        faqs = []
    
    charts = []
    
    # PREPROCESSING: Normalize question for better cache matching
    normalized_prompt = normalize_question(prompt)
    
    # SPECIAL CASE 1: Greeting detection (no LLM call needed)
    if is_greeting(prompt):
        logger.info(f"🎯 Detected greeting: {prompt}")
        return handle_greeting(prompt), []
    
    # SPECIAL CASE 2: Schema/Column questions
    if any(word in normalized_prompt for word in ['column', 'table', 'schema', 'show', 'list', 'columns', 'tables']):
        try:
            schema = get_database_schema(duckdb_conn)
            schema_response = format_schema_response(schema)
            return schema_response, []
        except Exception as e:
            logger.warning(f"Schema query failed: {str(e)}")
    
    # ✨ NEW STEP: Check if question matches a recommended FAQ
    matching_faq = find_matching_faq(normalized_prompt, faqs)
    if matching_faq:
        logger.info(f"📋 FAQ Match Found: {matching_faq['question']}")
        try:
            # Execute the pre-validated SQL
            result_df = duckdb_conn.query(matching_faq["sql"])
            if len(result_df) > 0:
                data_result = result_df.to_string()
                
                # Generate answer based on FAQ data
                answer_prompt = f"""FAQ Question: {matching_faq['question']}
Description: {matching_faq['description']}

Data Results:
```
{data_result}
```

Provide a clear, data-grounded answer based on these results.
Include specific metrics and numbers from the data.
Keep it concise (2-3 sentences)."""
                
                faq_response = call_llm(answer_prompt,
                    system_prompt="You are a fixed broadband analyst. Answer ONLY based on the data shown. Be specific with numbers and use EGP for currency.",
                    show_streaming=False,
                    use_cache=True,
                    db_context="")
                
                answer = faq_response.get("answer", "")
                
                # Format with FAQ indicator
                response_text = f"""✅ **FAQ Match: {matching_faq['question']}**

📊 **Data Retrieved:**
```
{data_result}
```

💡 **Analysis:**
{answer}"""
                
                logger.info(f"✅ FAQ-based answer generated")
                return response_text, charts
            
        except Exception as e:
            logger.error(f"FAQ query failed: {str(e)}")
            # Fall through to SQL-first approach
    
    # ✨ NEW: If no FAQ match, try semantic schema matching to identify relevant columns
    logger.info(f"🤖 No FAQ match, using semantic schema matching...")
    
    semantic_context = ""
    try:
        # Build column embeddings (cached)
        column_embeddings = build_column_embeddings(duckdb_conn)
        
        if column_embeddings:
            # Find most relevant columns for this question
            relevant_columns = find_relevant_columns(normalized_prompt, column_embeddings, top_k=3)
            
            if relevant_columns:
                relevant_cols_list = [col for col, score in relevant_columns]
                relevant_cols_str = ", ".join(relevant_cols_list)
                
                semantic_context = f"""
🎯 **SEMANTIC SCHEMA MATCHING** (Confidence Score: {relevant_columns[0][1]:.0%})
Most relevant columns for this question: {relevant_cols_str}

Focus on building a query using these columns."""
                
                logger.info(f"✅ Semantic context identified: {relevant_cols_str}")
            else:
                logger.info("⚠️ No similar columns found")
    except Exception as e:
        logger.warning(f"Semantic matching error: {str(e)}")
    
    # FALLBACK: If no FAQ match, use SQL-first approach with semantic context
    logger.info(f"🚀 Starting SQL-first approach...")
    
    try:
        # Build database context
        try:
            db_stats = duckdb_conn.query("""
                SELECT 
                    COUNT(*) as total_subscribers,
                    COUNT(DISTINCT Current_Package_Name) as package_types,
                    COUNT(DISTINCT Payment_Type) as payment_types,
                    COUNT(DISTINCT GOV) as governorates,
                    ROUND(AVG(Total_RPU), 2) as avg_revenue,
                    ROUND(MAX(Total_RPU), 2) as max_revenue,
                    ROUND(MIN(Total_RPU), 2) as min_revenue
                FROM broadband_subscribers
            """).df()
            stats_text = db_stats.iloc[0].to_dict()
            db_context = f"""🔒 CRITICAL - You MUST answer ONLY using this Egyptian fixed broadband database:

**Database:** broadband_subscribers - {int(stats_text['total_subscribers'])} fixed broadband subscriber records
**Avg Revenue:** {stats_text['avg_revenue']} EGP (Range: {stats_text['min_revenue']}-{stats_text['max_revenue']})
**Packages:** {int(stats_text['package_types'])} types | **Governorates:** {int(stats_text['governorates'])}

❌ NO external data, NO generic broadband trends, NO companies outside Egypt
✅ ONLY this database: {int(stats_text['total_subscribers'])} subscribers"""
        except:
            db_context = ""
        
        # STEP 1: Force LLM to construct SQL query FIRST (prevents hallucination)
        # Enhanced with semantic schema context if available
        sql_planning_prompt = f"""User Question: "{normalized_prompt}"
{semantic_context}

Write the EXACT SQL query to answer this question using broadband_subscribers table.
Columns available: subs_id, Fixed_Customer_No, ARPU, Current_Technology, Stability_Name, age, GOV, PopName, Gender, Tenure_Days, Subscriber_Status, Avg_Monthly_Payment, Total_RPU

**CRITICAL RULES:**
1. For string comparisons (Gender, Current_Technology, etc), use UPPER() for case-insensitive matching
   Example: WHERE UPPER(Gender) = 'FEMALE' instead of WHERE Gender = 'Female'
2. Use COUNT(*) for counting records
3. Always include meaningful aliases for results

Return ONLY the SQL query. Example format:
```sql
SELECT ... FROM giza_data ...
```"""
        
        sql_response = call_llm(sql_planning_prompt,
            system_prompt="You are a SQL expert. Write ONLY valid SQL, nothing else. No explanations.",
            show_streaming=False,
            use_cache=False,
            db_context=db_context)
        
        # Check both answer and reasoning fields (LLM may put SQL in reasoning)
        sql_text = (sql_response.get("answer") or sql_response.get("reasoning") or "").strip()
        
        # Extract SQL from markdown if present
        if "```sql" in sql_text:
            sql_text = sql_text.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_text:
            sql_text = sql_text.split("```")[1].split("```")[0].strip()
        
        # STEP 2: Execute the SQL query
        data_result = ""
        query_success = False
        query_error = ""
        
        if sql_text and len(sql_text) > 10 and "SELECT" in sql_text.upper():
            try:
                logger.info(f"📊 Executing SQL:\n{sql_text}")
                result_df = duckdb_conn.query(sql_text)
                data_result = result_df.to_string()
                query_success = True
                logger.info(f"✅ Query success: {len(result_df)} rows")
            except Exception as e:
                logger.error(f"❌ SQL Error: {str(e)}")
                query_error = str(e)
                query_success = False
                data_result = f"Query failed: {str(e)}"
        
        # STEP 3: Generate final answer based on query results with STREAMING
        if query_success and data_result and len(data_result) > 10:
            answer_prompt = f"""📊 **SQL Query Executed:**
```sql
{sql_text}
```

**Results:**
```
{data_result}
```

User Question: "{normalized_prompt}"

Based on the query results above:
1. State the specific numbers/metrics from the data
2. Provide analysis and insights  
3. Keep it concise (2-3 sentences)"""
            
            final_response = call_llm(answer_prompt,
                system_prompt="You are a fixed broadband analyst. Answer ONLY based on the query results shown. NO external data, NO generic advice, NO countries outside Egypt.",
                show_streaming=False,
                use_cache=True,
                db_context="")
            
            answer = final_response.get("answer", "")
            
            # Format response with query visibility
            response_text = f"""🔍 **SQL Query Executed:**
```sql
{sql_text}
```

📊 **Data Retrieved:**
```
{data_result}
```

✅ **Analysis:**
{answer}"""
            
            return response_text, charts
        
        else:
            # If SQL failed, return error message (don't call st.error, etc here)
            error_msg = f"⚠️ **Query Execution Failed:** {query_error}\n\nTry one of the recommended FAQ questions below." if query_error else "⚠️ That question is too specific. Try one of the recommended FAQ questions below."
            return {"error": True, "message": error_msg, "faqs": faqs[:5]}, charts
        
        # FALLBACK: Original understanding-based routing (if SQL fails)
        logger.info("📍 Fallback: Using understanding-based routing...")
        
        # STEP 1: LLM understands the question and identifies data needs (with caching)
        understanding_prompt = f"""Given this fixed broadband database (broadband_subscribers table with: Subs_ID, Subscriber_Status, Payment_Type, TTS_Total_Tickets, Count_Of_All_Outage, Current_Package_Name, Total_RPU, Tenure_Days, Gender, Has_GSM, GOV, Service_Number_Trimmed):

User asks: "{normalized_prompt}"

Identify: 
1. What type of analysis (ARPU/revenue, churn/risk, migration, demographics, correlation, etc.)
2. What columns/filters needed
3. Type of response expected"""
        
        understanding_result = call_llm(understanding_prompt, 
            system_prompt="You are a database analyst. Understand what data is needed. Be concise.",
            show_streaming=False,
            use_cache=True)  # Enable caching
        
        understanding = understanding_result.get("answer") or understanding_result.get("reasoning")
        logger.info(f"LLM Understanding: {understanding} (Cached: {understanding_result.get('cached', False)})")
        
        # STEP 2: Execute appropriate data query based on understanding
        data_summary = ""
        duckdb_tables = []
        try:
            duckdb_tables = duckdb_conn.list_tables()
        except:
            duckdb_tables = []
        
        if understanding and 'broadband_subscribers' in duckdb_tables:
            understanding_lower = understanding.lower()
            
            # Smart routing based on LLM understanding
            if any(word in understanding_lower for word in ['revenue', 'profit', 'income', 'total', 'payment']):
                try:
                    result = duckdb_conn.query("""
                        SELECT Payment_Type, Current_Package_Name, COUNT(*) as subscribers, 
                               ROUND(AVG(Total_RPU), 2) as avg_revenue_egp,
                               ROUND(SUM(Total_RPU), 2) as total_revenue_egp
                        FROM broadband_subscribers
                        WHERE Total_RPU IS NOT NULL AND Payment_Type IS NOT NULL
                        GROUP BY Payment_Type, Current_Package_Name
                        ORDER BY total_revenue_egp DESC
                    """).df()
                    data_summary = result.to_string()
                    logger.info(f"Revenue Query executed: {len(result)} rows")
                except Exception as e:
                    logger.error(f"Revenue query failed: {str(e)}")
                    data_summary = "No revenue data available"
            
            elif any(word in understanding_lower for word in ['outage', 'ticket', 'problem', 'issue', 'complaint', 'quality']):
                try:
                    result = duckdb_conn.query("""
                        SELECT Current_Package_Name, COUNT(*) as subscribers, 
                               ROUND(AVG(Count_Of_All_Outage), 2) as avg_outages,
                               ROUND(AVG(TTS_Total_Tickets), 2) as avg_tickets
                        FROM broadband_subscribers
                        WHERE Count_Of_All_Outage IS NOT NULL OR TTS_Total_Tickets IS NOT NULL
                        GROUP BY Current_Package_Name
                        ORDER BY avg_outages DESC
                    """).df()
                    data_summary = result.to_string()
                    logger.info(f"Service Quality Query executed: {len(result)} rows")
                except Exception as e:
                    logger.error(f"Service quality query failed: {str(e)}")
                    data_summary = "No service quality data available"
            
            elif any(word in understanding_lower for word in ['gender', 'demographic', 'population', 'gsm', 'cross-sell']):
                try:
                    result = duckdb_conn.query("""
                        SELECT Gender, Has_GSM, COUNT(*) as count, 
                               ROUND(AVG(Total_RPU), 2) as avg_revenue_egp
                        FROM broadband_subscribers
                        WHERE Gender IS NOT NULL OR Has_GSM IS NOT NULL
                        GROUP BY Gender, Has_GSM
                        ORDER BY count DESC
                        LIMIT 10
                    """).df()
                    data_summary = result.to_string()
                    logger.info(f"Demographics Query executed: {len(result)} rows")
                except Exception as e:
                    logger.error(f"Demographics query failed: {str(e)}")
                    data_summary = "No demographic data available"
            
            elif any(word in understanding_lower for word in ['package', 'plan', 'upgrade', 'tariff']):
                try:
                    result = duckdb_conn.query("""
                        SELECT Current_Package_Name, COUNT(*) as subscribers,
                               ROUND(AVG(Total_RPU), 2) as avg_revenue_egp,
                               ROUND(AVG(TTS_Total_Tickets), 2) as avg_tickets
                        FROM broadband_subscribers
                        WHERE Current_Package_Name IS NOT NULL
                        GROUP BY Current_Package_Name
                        ORDER BY subscribers DESC
                    """).df()
                    data_summary = result.to_string()
                    logger.info(f"Package Query executed: {len(result)} rows")
                except Exception as e:
                    logger.error(f"Package query failed: {str(e)}")
                    data_summary = "No package data available"
        
        # STEP 3: LLM generates final answer based on data with STREAMING & CACHING
        if data_summary and len(data_summary) > 10:
            final_prompt = f"""User Question: {normalized_prompt}

I found this data:
{data_summary}

Provide a direct answer based on this data:
1. Specific answer to their question
2. Key metrics and numbers
3. Business insights or next steps
Keep it concise (2-3 sentences)."""
            
            response_result = call_llm(final_prompt,
                system_prompt="You are a senior fixed broadband analyst. Give clear, specific answers ONLY based on the data provided. Use EGP for currency. Be concise. Do NOT make assumptions beyond the data.",
                show_streaming=False,
                use_cache=True,
                db_context=db_context)
            
            answer = response_result.get("answer")
            reasoning = response_result.get("reasoning")
            tokens = response_result.get("tokens", {})
            elapsed = response_result.get("time", {}).get("elapsed", 0)
            cached = response_result.get("cached", False)
            
            if answer and len(str(answer)) > 15:
                logger.info(f"✅ LLM Generated Answer (Cached: {cached})")
                
                # Format response with metrics
                response_text = f"🤖 **Answer:**\n\n{answer}"
                
                # Add metrics footer
                if tokens.get("total", 0) > 0 or elapsed > 0 or cached:
                    response_text += f"\n\n---\n📊 **Metrics:**"
                    if cached:
                        response_text += f"\n  • 💾 **Cached Response** (instant)"
                    if tokens.get("total", 0) > 0:
                        response_text += f"\n  • Tokens: {tokens.get('input', 0)} input + {tokens.get('output', 0)} output = {tokens.get('total', 0)} total"
                    if elapsed > 0 and not cached:
                        response_text += f"\n  • Time: {elapsed:.2f}s"
                
                return response_text, charts
        
        # Fallback: If LLM can't process, let LLM try generic response with streaming & caching
        generic_result = call_llm(normalized_prompt,
            system_prompt="You are a fixed broadband business analyst. Answer the user's question about subscriber data. Be helpful and specific.",
            show_streaming=False,
            use_cache=True)
        
        answer = generic_result.get("answer")
        tokens = generic_result.get("tokens", {})
        elapsed = generic_result.get("time", {}).get("elapsed", 0)
        cached = generic_result.get("cached", False)
        
        if answer and len(str(answer)) > 15:
            response_text = f"🤖 **Response:**\n\n{answer}"
            
            # Add metrics footer
            if tokens.get("total", 0) > 0 or elapsed > 0 or cached:
                response_text += f"\n\n---\n📊 **Metrics:**"
                if cached:
                    response_text += f"\n  • 💾 **Cached Response** (instant)"
                if tokens.get("total", 0) > 0:
                    response_text += f"\n  • Tokens: {tokens.get('input', 0)} input + {tokens.get('output', 0)} output = {tokens.get('total', 0)} total"
                if elapsed > 0 and not cached:
                    response_text += f"\n  • Time: {elapsed:.2f}s"
            
            return response_text, []
        else:
            return "📊 Please ask about ARPU, churn risk, technology, or demographics for detailed analysis.", []
    
    except Exception as e:
        logger.error(f"Router error: {str(e)}", exc_info=True)
        return f"⚠️ Error processing question: {str(e)}", []

# ==================== MAIN CONTENT ====================
if st.session_state.data is not None:
    df = st.session_state.data
    km = st.session_state.knowledge_manager
    
    # Dashboard Overview
    st.subheader("📊 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Subscribers", f"{len(df):,}")
    with col2:
        avg_revenue = df['Total_RPU'].mean() if 'Total_RPU' in df.columns else 0
        st.metric("Avg Revenue (EGP)", f"{avg_revenue:.0f} EGP")
    with col3:
        if 'Subscriber_Status' in df.columns:
            active = (df['Subscriber_Status'] == 'Active').sum()
            st.metric("Active Subscribers", f"{active:,}")
        else:
            st.metric("Active Subscribers", "N/A")
    with col4:
        if 'Has_GSM' in df.columns:
            gsm = (df['Has_GSM'] == 'Yes').sum()
            st.metric("GSM Cross-sell", f"{gsm:,}")
        else:
            st.metric("GSM Cross-sell", "N/A")
    
    st.markdown("---")
    
    # ==================== CHAT INPUT (Outside tabs - required by Streamlit) ====================
    st.subheader("💬 Ask the Agent")
    
    # Get chat input at main level (NOT inside tabs/expanders)
    user_prompt = st.chat_input("Ask about ARPU, churn, technology, demographics... or select a FAQ below")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Agent Chat",
        "📊 Analytics Dashboard",
        "📋 Data Explorer",
        "💡 Knowledge Base"
    ])
    
    # ===================== TAB 1: AGENT CHAT =====================
    with tab1:
        st.subheader("💬 Intelligent Data Analyst Agent")
        
        # Display recommended FAQs in expandable section
        faq_section = st.expander("📋 **Recommended Questions (Always Work!)**", expanded=False)
        with faq_section:
            st.markdown("These questions are guaranteed to work and won't cause hallucination:")
            cols = st.columns(2)
            for i, faq in enumerate(st.session_state.recommended_faqs[:6]):
                with cols[i % 2]:
                    if st.button(faq['question'], key=f"faq_{i}"):
                        # Simulate user input by setting chat input
                        st.session_state.user_selected_faq = faq['question']
                        st.rerun()
        
        st.markdown("---")
        
        # Display chat history (organized and clean)
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    # Check if message is a dict (has error/faqs) or string
                    if isinstance(msg["content"], dict):
                        content = msg["content"]
                        if content.get("error"):
                            st.error(content.get("message", "An error occurred"))
                            if content.get("faqs"):
                                st.info("💡 Try one of these recommended questions:")
                                for faq in content.get("faqs", []):
                                    st.caption(f"• {faq['question']}")
                        else:
                            st.markdown(content.get("message", ""))
                    else:
                        # Plain string response
                        st.markdown(msg["content"])
        
        # Agent context for future use
        agent_context = km.build_agent_context()
    
    # ==================== PROCESS USER INPUT (Handle both direct input and FAQ selection) ====================
    # Check if user selected an FAQ
    if "user_selected_faq" in st.session_state and st.session_state.user_selected_faq:
        user_prompt = st.session_state.user_selected_faq
        st.session_state.user_selected_faq = None
    
    if user_prompt:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        
        # Display in chat tab
        with tab1:
            # Show user message
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                with response_placeholder.container():
                    with st.spinner("🤖 Analyzing question..."):
                        try:
                            # Route question with FAQs to prevent hallucination
                            response, charts = _route_question(user_prompt, st.session_state.tools, km, st.session_state.duckdb, st.session_state.recommended_faqs)
                            
                            # Handle dictionary responses (errors with FAQs)
                            if isinstance(response, dict):
                                if response.get("error"):
                                    st.error(response.get("message"))
                                    if response.get("faqs"):
                                        st.info("💡 Try one of these recommended questions:")
                                        for faq in response.get("faqs", []):
                                            st.caption(f"• **{faq['question']}**  \n_{faq['description']}_")
                                response_str = response.get("message", "")
                            else:
                                # Plain string response
                                response_str = response
                                st.markdown(response_str)
                            
                            # Render visualizations if available
                            if charts:
                                st.markdown("---")
                                st.subheader("📊 Visualizations")
                                chart_cols = st.columns(len(charts))
                                for idx, (title, fig) in enumerate(charts):
                                    with chart_cols[idx % len(chart_cols)]:
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            # Add to history (store as string or dict)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            
                            # Log the interaction
                            logger.info(f"✅ Question: {user_prompt}")
                            logger.info(f"✅ Response generated with {len(charts)} visualizations")
                            
                        except Exception as e:
                            error_msg = f"❌ Analysis failed: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                            logger.error(f"Chat error: {str(e)}", exc_info=True)
    
    # ===================== TAB 2: ANALYTICS DASHBOARD =====================
    with tab2:
        st.subheader("📈 Multi-dimensional Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ARPU by Technology
            if 'Current_Technology' in df.columns and 'ARPU' in df.columns:
                try:
                    tech_arpu = df.groupby('Current_Technology').agg({
                        'ARPU': ['count', 'mean', 'sum']
                    }).reset_index()
                    tech_arpu.columns = ['Technology', 'Count', 'Avg ARPU', 'Total Revenue']
                    tech_arpu = tech_arpu.sort_values('Avg ARPU', ascending=False)
                    
                    fig = px.bar(
                        tech_arpu,
                        x='Technology',
                        y='Avg ARPU',
                        color='Avg ARPU',
                        title='ARPU by Technology',
                        text='Avg ARPU'
                    )
                    fig.update_traces(texttemplate='₵%{text:.0f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")
        
        with col2:
            # Stability Distribution
            if 'Stability_Name' in df.columns:
                try:
                    stability = df['Stability_Name'].value_counts()
                    fig = px.pie(
                        values=stability.values,
                        names=stability.index,
                        title='Subscriber Stability Distribution',
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # ARPU by Stability
            if 'Stability_Name' in df.columns and 'ARPU' in df.columns:
                try:
                    stability_arpu = df.groupby('Stability_Name')['ARPU'].mean().sort_values(ascending=False)
                    fig = px.bar(
                        x=stability_arpu.index,
                        y=stability_arpu.values,
                        title='ARPU by Subscriber Stability',
                        labels={'x': 'Stability', 'y': 'Avg ARPU (EGP)'},
                        text=stability_arpu.values
                    )
                    fig.update_traces(texttemplate='₵%{text:.0f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")
        
        with col4:
            # Population Distribution
            if 'PopName' in df.columns:
                try:
                    pop = df['PopName'].value_counts()
                    fig = px.pie(
                        values=pop.values,
                        names=pop.index,
                        title='Population Segment Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")
    
    # ===================== TAB 3: DATA EXPLORER =====================
    with tab3:
        st.subheader("🔍 Raw Data Explorer")
        
        # Column selector - only use defaults that actually exist in dataframe
        available_cols = df.columns.tolist()
        default_cols = ['subs_id', 'ARPU', 'Current_Technology', 'Stability_Name', 'age']
        safe_defaults = [col for col in default_cols if col in available_cols]
        
        columns = st.multiselect(
            "Select columns to display",
            options=available_cols,
            default=safe_defaults if safe_defaults else available_cols[:5]
        )
        
        if columns:
            st.dataframe(
                df[columns].head(100),
                use_container_width=True,
                height=400
            )
        
        # Download option
        st.download_button(
            label="📥 Download Data (CSV)",
            data=df.to_csv(index=False),
            file_name=f"broadband_subscribers_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
        )
    
    # ===================== TAB 4: KNOWLEDGE BASE =====================
    with tab4:
        st.subheader("📚 Knowledge Base & Learnings")
        
        kb_col1, kb_col2 = st.columns(2)
        
        with kb_col1:
            st.subheader("📊 Table Schemas")
            for table_name in km.tables.keys():
                with st.expander(f"📋 {table_name}"):
                    schema = km.tables[table_name]
                    st.write(schema.get('description', 'N/A'))
                    st.json({k: v for k, v in schema.get('columns', {}).items()})
        
        with kb_col2:
            st.subheader("🎯 Business Metrics")
            metrics_dict = km.business_rules.get('broadband_metrics', {}).get('metrics', {})
            for metric_name, metric_def in metrics_dict.items():
                with st.expander(f"📈 {metric_name}"):
                    st.write(f"**Definition:** {metric_def.get('business_value', 'N/A')}")
                    st.write(f"**Unit:** {metric_def.get('unit', 'N/A')}")
        
        st.subheader("💡 Discovered Insights")
        insights = km.get_business_insights()
        for insight in insights[:5]:
            with st.expander(f"🔍 {insight.get('insight', 'N/A')}"):
                st.write(f"**Confidence:** {insight.get('confidence', 'N/A')}")
                st.write(f"**Recommendation:** {insight.get('recommendation', 'N/A')}")

else:
    st.info("👈 **Click 'Initialize Agent' to load Fixed Broadband subscriber data from Teradata**")
    st.markdown("""
    ## 🎯 What This Agent Does:
    
    1. **Connects to Teradata** - Fetches latest Giza subscriber data
    2. **Loads to DuckDB** - Ultra-fast in-memory analytics
    3. **Leverages Knowledge Base**:
       - 📊 Table schemas & relationships
       - 📋 Business metrics & definitions
       - 🔍 Query patterns that work
       - 💡 Discovered insights
    4. **Answers Natural Language Questions**:
       - "What are ARPU trends?"
       - "Which technology segment is most profitable?"
       - "Who are the at-risk high-value customers?"
    
    ## 🚀 Key Features:
    - **Layered Context**: Integrates tables, business rules, queries, learnings
    - **Intelligent Tools**: ARPU analysis, churn risk, FTTH migration, demographics
    - **Data Quality**: Auto-cleaning, deduplication, standardization
    - **Knowledge Learning**: Captures patterns and builds institutional memory
    """)

# ==================== HELPER FUNCTIONS ====================


if __name__ == "__main__":
    logger.info("🚀 Telecom Analytics Agent started")
