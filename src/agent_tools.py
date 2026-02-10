"""
Custom Tools for Agno Agent - Telecom Analytics
"""
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from src.connectors import DuckDBConnector
from src.knowledge_manager import KnowledgeManager
from src.visualizations import VisualizationManager, ChartDetector

logger = logging.getLogger(__name__)


class TelecomAnalyticsTool:
    """Base class for telecom analytics tools"""
    
    def __init__(self, duckdb_conn: DuckDBConnector, knowledge_manager: KnowledgeManager):
        self.duckdb = duckdb_conn
        self.km = knowledge_manager
        self.viz_manager = VisualizationManager()
    
    def add_visualization_metadata(self, result: Dict[str, Any], 
                                   question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhance result with visualization suggestions"""
        try:
            viz_suggestions = self.viz_manager.suggest_visualizations(question, df, result)
            result['visualization'] = {
                'recommended': len(viz_suggestions.get('charts', [])) > 0,
                'primary_type': viz_suggestions.get('primary_chart'),
                'chart_count': len(viz_suggestions.get('charts', [])),
                'recommendation': viz_suggestions.get('recommendation'),
                'sample_data': df.head(5).to_dict('records') if len(df) > 0 else []
            }
        except Exception as e:
            logger.warning(f"Visualization metadata generation failed: {str(e)}")
            result['visualization'] = {'recommended': False}
        return result


class ARPUAnalysisTool(TelecomAnalyticsTool):
    """Analyze ARPU metrics by various dimensions"""
    
    def by_technology(self) -> Dict[str, Any]:
        """Get ARPU breakdown by technology"""
        query = """
        SELECT 
            Current_Technology,
            COUNT(*) as subscribers,
            ROUND(AVG(ARPU), 2) as avg_arpu,
            ROUND(SUM(ARPU), 2) as total_revenue
        FROM giza_data
        GROUP BY 1
        ORDER BY 3 DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "ARPU by Technology",
            "data": df.to_dict('records') if df is not None else [],
            "insight": "Shows which technologies generate highest revenue per user"
        }
    
    def by_stability(self) -> Dict[str, Any]:
        """Get ARPU breakdown by subscriber stability"""
        query = """
        SELECT 
            Stability_Name,
            COUNT(*) as subscribers,
            ROUND(AVG(ARPU), 2) as avg_arpu,
            ROUND(AVG(Tenure_Days), 0) as avg_tenure_days
        FROM giza_data
        WHERE Stability_Name IS NOT NULL
        GROUP BY 1
        ORDER BY 3 DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "ARPU by Stability",
            "data": df.to_dict('records') if df is not None else [],
            "insight": "Stable customers generate more revenue"
        }
    
    def by_geography(self) -> Dict[str, Any]:
        """Get ARPU breakdown by geography"""
        query = """
        SELECT 
            GOV,
            Count(*) as subscribers,
            ROUND(AVG(ARPU), 2) as avg_arpu
        FROM giza_data
        GROUP BY 1
        ORDER BY 3 DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "ARPU by Geography",
            "data": df.to_dict('records') if df is not None else [],
            "insight": "Geographic performance comparison"
        }


class ChurnRiskTool(TelecomAnalyticsTool):
    """Analyze churn risk and at-risk segments"""
    
    def risk_distribution(self) -> Dict[str, Any]:
        """Get distribution of subscribers by risk level"""
        query = """
        SELECT 
            Stability_Name,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct,
            ROUND(AVG(ARPU), 2) as avg_arpu
        FROM giza_data
        WHERE Stability_Name IS NOT NULL
        GROUP BY 1
        ORDER BY count DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "Risk Distribution",
            "data": df.to_dict('records') if df is not None else [],
            "alerts": self._generate_risk_alerts(df)
        }
    
    def at_risk_segments(self) -> Dict[str, Any]:
        """Identify at-risk customer segments for retention"""
        query = """
        SELECT 
            Stability_Name,
            Current_Technology,
            COUNT(*) as customers,
            ROUND(AVG(ARPU), 2) as revenue_at_risk,
            ROUND(SUM(ARPU), 2) as total_revenue_at_risk
        FROM giza_data
        WHERE Stability_Name IN ('At-Risk', 'Churner')
        GROUP BY 1, 2
        ORDER BY 5 DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "At-Risk Revenue",
            "data": df.to_dict('records') if df is not None else [],
            "recommendation": "Target retention campaigns to highest revenue at-risk segments"
        }
    
    def _generate_risk_alerts(self, df: Optional[pd.DataFrame]) -> List[str]:
        """Generate alerts based on risk distribution"""
        alerts = []
        if df is None or df.empty:
            return alerts
        
        at_risk = df[df['Stability_Name'] == 'At-Risk']
        if not at_risk.empty and at_risk['pct'].values[0] > 30:
            alerts.append("⚠️ ALERT: Over 30% of subscriber base at risk of churn")
        
        return alerts


class TechnologyMigrationTool(TelecomAnalyticsTool):
    """Identify technology migration opportunities"""
    
    def ftth_migration_potential(self) -> Dict[str, Any]:
        """Calculate FTTH migration opportunity"""
        query = """
        SELECT 
            COUNT(*) as migration_candidates,
            ROUND(AVG(ARPU), 2) as current_avg_arpu,
            ROUND(AVG(ARPU) * 1.4, 2) as potential_avg_arpu,
            ROUND(SUM(ARPU), 2) as current_revenue,
            ROUND(SUM(ARPU) * 1.4, 2) as potential_revenue,
            ROUND(SUM(ARPU) * 0.4, 2) as revenue_uplift
        FROM giza_data
        WHERE Current_Technology != 'FTTH'
          AND ARPU > 100
          AND Stability_Name = 'Stable'
        """
        df = self.duckdb.query(query)
        if df is not None and not df.empty:
            data = df.to_dict('records')[0]
        else:
            data = {}
        
        return {
            "metric": "FTTH Migration Opportunity",
            "data": data,
            "recommendation": "FTTH upgrade drives 40% ARPU increase - prioritize stable, high-value customers"
        }


class DemographicAnalysisTool(TelecomAnalyticsTool):
    """Analyze subscriber demographics and segments"""
    
    def by_age_group(self) -> Dict[str, Any]:
        """Segment subscribers by age groups"""
        query = """
        SELECT 
            CASE 
                WHEN age < 25 THEN '18-24'
                WHEN age < 35 THEN '25-34'
                WHEN age < 50 THEN '35-49'
                WHEN age < 65 THEN '50-64'
                ELSE '65+'
            END as age_group,
            COUNT(*) as subscribers,
            ROUND(AVG(ARPU), 2) as avg_arpu,
            ROUND(AVG(Tenure_Days), 0) as avg_tenure_days
        FROM giza_data
        GROUP BY 1
        ORDER BY 3 DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "Demographics by Age",
            "data": df.to_dict('records') if df is not None else [],
            "insight": "Identifies premium age segments for targeted marketing"
        }
    
    def by_population_type(self) -> Dict[str, Any]:
        """Analyze Urban/Suburban/Rural segments"""
        query = """
        SELECT 
            PopName,
            COUNT(*) as subscribers,
            ROUND(AVG(ARPU), 2) as avg_arpu,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct_of_total
        FROM giza_data
        GROUP BY 1
        ORDER BY 3 DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "Demographics by Area",
            "data": df.to_dict('records') if df is not None else [],
            "insight": "Identifies growth potential in under-served areas"
        }


class ReportGeneratorTool(TelecomAnalyticsTool):
    """Generate structured business intelligence reports"""
    
    def executive_summary(self) -> Dict[str, Any]:
        """Generate executive dashboard snapshot"""
        
        # Total subscribers
        total_query = "SELECT COUNT(*) as total, ROUND(AVG(ARPU), 2) as avg_arpu FROM giza_data"
        total = self.duckdb.query(total_query)
        
        # Risk breakdown
        risk_query = """
        SELECT Stability_Name, COUNT(*) as count
        FROM giza_data
        WHERE Stability_Name IS NOT NULL
        GROUP BY 1
        """
        risk = self.duckdb.query(risk_query)
        
        # Tech breakdown
        tech_query = """
        SELECT Current_Technology, COUNT(*) as count
        FROM giza_data
        GROUP BY 1
        ORDER BY count DESC
        LIMIT 3
        """
        tech = self.duckdb.query(tech_query)
        
        return {
            "title": "Executive Summary",
            "total_customers": total.to_dict('records')[0] if total is not None else {},
            "risk_distribution": risk.to_dict('records') if risk is not None else [],
            "top_technologies": tech.to_dict('records') if tech is not None else []
        }
    
    def dashboard_with_visualizations(self) -> Dict[str, Any]:
        """Generate dashboard with visualization recommendations"""
        
        # Fetch data for all key metrics
        tech_arpu = self.duckdb.query("""
            SELECT Current_Technology, COUNT(*) as count, AVG(ARPU) as avg_arpu
            FROM giza_data GROUP BY 1 ORDER BY avg_arpu DESC
        """)
        
        risk_dist = self.duckdb.query("""
            SELECT Stability_Name, COUNT(*) as count
            FROM giza_data WHERE Stability_Name IS NOT NULL GROUP BY 1
        """)
        
        demo_age = self.duckdb.query("""
            SELECT 
                CASE WHEN age < 25 THEN '18-24'
                     WHEN age < 35 THEN '25-34'
                     WHEN age < 50 THEN '35-49'
                     WHEN age < 65 THEN '50-64'
                     ELSE '65+' END as age_group,
                COUNT(*) as count
            FROM giza_data GROUP BY 1
        """)
        
        dashboard = {
            "title": "Intelligence Dashboard",
            "timestamp": pd.Timestamp.now().isoformat(),
            "panels": []
        }
        
        # Panel 1: ARPU by Technology
        if tech_arpu is not None and not tech_arpu.empty:
            panel1 = {
                "title": "ARPU by Technology",
                "data": tech_arpu.to_dict('records'),
                "visualization": {
                    "type": "bar",
                    "recommendation": "Bar chart shows technology profitability ranking",
                    "key_insight": f"{tech_arpu.iloc[0]['Current_Technology']} leads with ₵{tech_arpu.iloc[0]['avg_arpu']:.0f} avg ARPU"
                }
            }
            dashboard["panels"].append(panel1)
        
        # Panel 2: Risk Distribution
        if risk_dist is not None and not risk_dist.empty:
            panel2 = {
                "title": "Subscriber Stability",
                "data": risk_dist.to_dict('records'),
                "visualization": {
                    "type": "pie",
                    "recommendation": "Pie chart shows composition of risk segments",
                    "key_insight": f"{risk_dist.iloc[0]['Stability_Name']}: {risk_dist.iloc[0]['count']:,} subscribers"
                }
            }
            dashboard["panels"].append(panel2)
        
        # Panel 3: Demographics
        if demo_age is not None and not demo_age.empty:
            panel3 = {
                "title": "Age Distribution",
                "data": demo_age.to_dict('records'),
                "visualization": {
                    "type": "bar",
                    "recommendation": "Bar chart shows demographic concentration",
                    "key_insight": f"Largest segment: {demo_age.iloc[0]['age_group']} ({demo_age.iloc[0]['count']:,})"
                }
            }
            dashboard["panels"].append(panel3)
        
        return dashboard
    
    def full_analytics_report(self) -> str:
        """Generate comprehensive JSON report"""
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "sections": {
                "executive_summary": self.executive_summary(),
                "arpu_analysis": ARPUAnalysisTool(self.duckdb, self.km).by_technology(),
                "churn_analysis": ChurnRiskTool(self.duckdb, self.km).risk_distribution(),
                "migration_opportunity": TechnologyMigrationTool(self.duckdb, self.km).ftth_migration_potential(),
                "dashboard": self.dashboard_with_visualizations()
            }
        }
        return json.dumps(report, indent=2)


def create_tool_functions(duckdb_conn: DuckDBConnector, km: KnowledgeManager) -> Dict:
    """Create callable tool functions for Agno agent"""
    
    arpu_tool = ARPUAnalysisTool(duckdb_conn, km)
    churn_tool = ChurnRiskTool(duckdb_conn, km)
    tech_tool = TechnologyMigrationTool(duckdb_conn, km)
    demo_tool = DemographicAnalysisTool(duckdb_conn, km)
    report_tool = ReportGeneratorTool(duckdb_conn, km)
    
    return {
        "arpu_by_technology": lambda: json.dumps(arpu_tool.by_technology()),
        "arpu_by_stability": lambda: json.dumps(arpu_tool.by_stability()),
        "arpu_by_geography": lambda: json.dumps(arpu_tool.by_geography()),
        "churn_risk_distribution": lambda: json.dumps(churn_tool.risk_distribution()),
        "at_risk_segments": lambda: json.dumps(churn_tool.at_risk_segments()),
        "ftth_migration_analysis": lambda: json.dumps(tech_tool.ftth_migration_potential()),
        "demographics_by_age": lambda: json.dumps(demo_tool.by_age_group()),
        "demographics_by_area": lambda: json.dumps(demo_tool.by_population_type()),
        "executive_summary": lambda: json.dumps(report_tool.executive_summary()),
        "intelligence_dashboard": lambda: json.dumps(report_tool.dashboard_with_visualizations()),
        "full_analytics_report": report_tool.full_analytics_report
    }
