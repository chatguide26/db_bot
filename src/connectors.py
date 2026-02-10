"""
Teradata and DuckDB Connectors with Intelligent Query Execution
"""
import pandas as pd
import duckdb
import teradatasql
import logging
from typing import Optional, List, Dict, Any
import os

logger = logging.getLogger(__name__)


class TeradataConnector:
    """Secure Teradata connection handler"""
    
    def __init__(self, host: str, username: str, password: str, database: str):
        self.host = host
        self.username = username
        self.password = password
        self.database = database
        self.conn = None
    
    def connect(self):
        """Establish Teradata connection"""
        try:
            self.conn = teradatasql.connect(
                host=self.host,
                user=self.username,
                password=self.password,
                database=self.database,
                dbs_port=1025,
                tmode="ANSI"
            )
            logger.info(f"✅ Connected to Teradata: {self.host}/{self.database}")
            return True
        except Exception as e:
            logger.error(f"❌ Teradata connection failed: {str(e)}")
            return False
    
    def query(self, sql: str, params: Optional[List] = None) -> Optional[pd.DataFrame]:
        """Execute query and return DataFrame"""
        if not self.conn:
            logger.error("No Teradata connection")
            return None
        
        try:
            df = pd.read_sql(sql, self.conn, params=params)
            logger.info(f"✅ Query executed: {len(df):,} rows")
            return df
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}")
            return None
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            logger.info("Teradata connection closed")


class DuckDBConnector:
    """In-memory DuckDB analytics engine"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
    
    def load_dataframe(self, df: pd.DataFrame, table_name: str):
        """Load DataFrame into DuckDB table"""
        try:
            self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
            logger.info(f"✅ Loaded {len(df):,} rows into DuckDB table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load DataFrame: {str(e)}")
            return False
    
    def query(self, sql: str) -> Optional[pd.DataFrame]:
        """Execute DuckDB query"""
        try:
            result = self.conn.execute(sql).fetchdf()
            logger.info(f"✅ DuckDB query returned {len(result):,} rows")
            return result
        except Exception as e:
            logger.error(f"❌ DuckDB query failed: {str(e)}")
            return None
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get table statistics for context"""
        try:
            stats_sql = f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT subs_id) as unique_subscribers
            FROM {table_name}
            """
            result = self.conn.execute(stats_sql).fetchone()
            return {
                'total_rows': result[0],
                'unique_subscribers': result[1]
            }
        except:
            return {}
    
    def close(self):
        """Close connection"""
        self.conn.close()


class SmartDataProcessor:
    """Intelligent data cleaning and standardization"""
    
    @staticmethod
    def preprocess_subscriber_data(df: pd.DataFrame) -> pd.DataFrame:
        """Apply telecom-specific data cleaning"""
        
        # Fix date columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Standardize technology names
        if 'Current_Technology' in df.columns:
            tech_mapping = {
                'Fiber': 'FTTH',
                'FTT': 'FTTH',
                'LTE-A': 'LTE',
                '4GLT': 'LTE'
            }
            df['Current_Technology'] = df['Current_Technology'].replace(tech_mapping)
        
        # Fix numeric columns
        numeric_cols = df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            if df[col].dtype == 'object' and df[col].name in ['ARPU', 'Total_RPU', 'Avg_Monthly_Payment']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove duplicates by subs_id (keep latest)
        if 'subs_id' in df.columns and 'Insertion_Date' in df.columns:
            df = df.sort_values('Insertion_Date', ascending=False).drop_duplicates(
                subset=['subs_id'], keep='first'
            )
        
        # Remove rows with critical nulls
        critical_cols = ['subs_id', 'ARPU', 'Stability_Name']
        df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
        
        logger.info(f"✅ Data preprocessing complete: {len(df):,} clean rows")
        return df
