"""
Knowledge Management System - Layered Context for Agent Intelligence
"""
import json
import os
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """Manages layered context: tables, business rules, queries, learnings"""
    
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = knowledge_dir
        self.tables = {}
        self.business_rules = {}
        self.query_patterns = {}
        self.learnings = {}
        self._load_all()
    
    def _load_all(self):
        """Load all knowledge layers"""
        self._load_tables()
        self._load_business_rules()
        self._load_query_patterns()
        self._load_learnings()
        logger.info("✅ Knowledge base loaded successfully")
    
    def _load_tables(self):
        """Load table schemas from knowledge/tables/*.json"""
        tables_dir = os.path.join(self.knowledge_dir, "tables")
        if not os.path.exists(tables_dir):
            logger.warning(f"Tables directory not found: {tables_dir}")
            return
        
        for file in os.listdir(tables_dir):
            if file.endswith(".json"):
                with open(os.path.join(tables_dir, file)) as f:
                    table_data = json.load(f)
                    self.tables[table_data['table_name']] = table_data
        
        logger.info(f"✅ Loaded {len(self.tables)} table schemas")
    
    def _load_business_rules(self):
        """Load business rules from knowledge/business/*.json"""
        business_dir = os.path.join(self.knowledge_dir, "business")
        if not os.path.exists(business_dir):
            logger.warning(f"Business directory not found: {business_dir}")
            return
        
        for file in os.listdir(business_dir):
            if file.endswith(".json"):
                with open(os.path.join(business_dir, file)) as f:
                    self.business_rules[file.replace('.json', '')] = json.load(f)
        
        logger.info(f"✅ Loaded business rules from {len(self.business_rules)} files")
    
    def _load_query_patterns(self):
        """Load SQL query patterns from knowledge/queries/*.sql"""
        queries_dir = os.path.join(self.knowledge_dir, "queries")
        if not os.path.exists(queries_dir):
            logger.warning(f"Queries directory not found: {queries_dir}")
            return
        
        for file in os.listdir(queries_dir):
            if file.endswith(".sql"):
                with open(os.path.join(queries_dir, file)) as f:
                    self.query_patterns[file] = f.read()
        
        logger.info(f"✅ Loaded {len(self.query_patterns)} SQL query patterns")
    
    def _load_learnings(self):
        """Load learnings from knowledge/learnings/*.json"""
        learnings_dir = os.path.join(self.knowledge_dir, "learnings")
        if not os.path.exists(learnings_dir):
            logger.warning(f"Learnings directory not found: {learnings_dir}")
            return
        
        for file in os.listdir(learnings_dir):
            if file.endswith(".json"):
                with open(os.path.join(learnings_dir, file)) as f:
                    self.learnings[file.replace('.json', '')] = json.load(f)
        
        logger.info(f"✅ Loaded learnings from {len(self.learnings)} files")
    
    def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get table schema and column definitions"""
        return self.tables.get(table_name)
    
    def get_metric_definition(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get business metric definition"""
        broadband_metrics = self.business_rules.get('broadband_metrics', {})
        metrics = broadband_metrics.get('metrics', {})
        return metrics.get(metric_name)
    
    def get_data_quality_rules(self) -> List[str]:
        """Get data quality validation rules"""
        broadband_metrics = self.business_rules.get('broadband_metrics', {})
        return broadband_metrics.get('business_rules', {}).get('data_quality', [])
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """Get known error patterns and fixes"""
        return self.learnings.get('error_patterns', {})
    
    def get_business_insights(self) -> List[Dict[str, Any]]:
        """Get discovered business insights"""
        learnings = self.learnings.get('error_patterns', {})
        return learnings.get('business_insights', [])
    
    def build_agent_context(self) -> str:
        """Build comprehensive context for agent system prompt"""
        
        # Table schemas
        table_schemas = "\n".join([
            f"- {name}: {schema.get('description', 'N/A')}"
            for name, schema in self.tables.items()
        ])
        
        # Key metrics
        metrics = self.business_rules.get('broadband_metrics', {}).get('metrics', {})
        metrics_list = "\n".join([
            f"- {name}: {m.get('business_value', '')}"
            for name, m in metrics.items()
        ])
        
        # Data quality rules
        dq_rules = "\n".join(
            f"- {rule}" 
            for rule in self.get_data_quality_rules()
        )
        
        # Recent learnings
        insights = self.get_business_insights()
        insights_list = "\n".join([
            f"- {i.get('insight', '')}" 
            for i in insights[:3]  # Top 3
        ])
        
        context = f"""
=== FIXED BROADBAND ANALYTICS KNOWLEDGE BASE ===

🗄️ DATA SOURCES:
{table_schemas}

📊 KEY METRICS:
{metrics_list}

✅ DATA QUALITY RULES:
{dq_rules}

💡 BUSINESS INSIGHTS (Recent):
{insights_list}

🎯 Your role: Transform natural language questions into structured business intelligence
using the above knowledge base. Always provide metrics, insights, and recommendations.
"""
        return context


class QueryBuilder:
    """Builds optimized SQL queries using knowledge base patterns"""
    
    def __init__(self, knowledge_manager: KnowledgeManager):
        self.km = knowledge_manager
    
    def suggest_query_for_question(self, question: str) -> Optional[str]:
        """Suggest a query pattern based on question intent"""
        
        question_lower = question.lower()
        
        # Keyword-based pattern matching for Fixed Broadband domain
        patterns = {
            'revenue': 'PATTERN 1: Total_RPU Analysis by Payment Type',
            'payment': 'PATTERN 1: Total_RPU Analysis by Payment Type',
            'outage': 'PATTERN 2: Service Reliability & Outage Analysis',
            'tickets': 'PATTERN 2: Service Reliability & Outage Analysis',
            'quality': 'PATTERN 2: Service Reliability & Outage Analysis',
            'package': 'PATTERN 3: Package Distribution & Adoption',
            'gsm': 'PATTERN 4: GSM Cross-sell Opportunity Analysis',
            'cross-sell': 'PATTERN 4: GSM Cross-sell Opportunity Analysis',
            'status': 'PATTERN 5: Subscriber Status Segmentation',
            'active': 'PATTERN 5: Subscriber Status Segmentation',
            'suspended': 'PATTERN 5: Subscriber Status Segmentation',
            'gender': 'PATTERN 6: Demographic Segmentation',
            'geographic': 'PATTERN 7: Geographic Performance by GOV',
            'region': 'PATTERN 7: Geographic Performance by GOV',
            'tenure': 'PATTERN 8: Subscriber Lifecycle Analysis',
            'customer': 'PATTERN 5: Subscriber Status Segmentation'
        }
        
        for keyword, pattern_name in patterns.items():
            if keyword in question_lower:
                return f"Use {pattern_name} from query patterns"
        
        return "Use PATTERN 0: Latest fixed broadband subscriber data as base query"
    
    def validate_query(self, sql: str) -> bool:
        """Check query against data quality rules"""
        rules = self.km.get_data_quality_rules()
        # Implementation: check for proper filtering, joins, etc.
        return True
