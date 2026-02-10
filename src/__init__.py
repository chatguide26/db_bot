"""
Telecom Analytics Intelligence Agent
AI-Powered Business Intelligence for Telecom Operators
"""

__version__ = "1.0.0"
__author__ = "Analytics Team"
__description__ = "Agno Agent + Teradata + DuckDB + Knowledge Base"

from src.connectors import TeradataConnector, DuckDBConnector, SmartDataProcessor
from src.knowledge_manager import KnowledgeManager, QueryBuilder
from src.agent_tools import create_tool_functions

__all__ = [
    'TeradataConnector',
    'DuckDBConnector', 
    'SmartDataProcessor',
    'KnowledgeManager',
    'QueryBuilder',
    'create_tool_functions'
]
