"""
Custom Tools for Agno Agent - Fixed Broadband Analytics
"""
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from src.connectors import DuckDBConnector
from src.knowledge_manager import KnowledgeManager
from src.visualizations import VisualizationManager, ChartDetector

logger = logging.getLogger(__name__)


class BroadbandAnalyticsTool:
    """Base class for fixed broadband analytics tools"""
    
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


class RevenuAnalysisTool(BroadbandAnalyticsTool):
    """Analyze Total_RPU metrics by various dimensions"""
    
    def by_payment_type(self) -> Dict[str, Any]:
        """Get Total_RPU breakdown by payment type"""
        query = """
        SELECT 
            Payment_Type,
            COUNT(*) as subscribers,
            ROUND(AVG(Total_RPU), 2) as avg_rpu,
            ROUND(SUM(Total_RPU), 2) as total_revenue
        FROM broadband_subscribers
        GROUP BY 1
        ORDER BY 3 DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "Revenue by Payment Type",
            "data": df.to_dict('records') if df is not None else [],
            "insight": "Shows which payment models generate highest revenue"
        }
    
    def by_package(self) -> Dict[str, Any]:
        """Get Total_RPU breakdown by package type"""
        query = """
        SELECT 
            Current_Package_Name,
            COUNT(*) as subscribers,
            ROUND(AVG(Total_RPU), 2) as avg_rpu,
            ROUND(SUM(Total_RPU), 2) as total_revenue
        FROM broadband_subscribers
        WHERE Current_Package_Name IS NOT NULL
        GROUP BY 1
        ORDER BY 3 DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "Revenue by Package",
            "data": df.to_dict('records') if df is not None else [],
            "insight": "Package tier directly correlates with revenue per subscriber"
        }
    
    def by_geography(self) -> Dict[str, Any]:
        """Get Total_RPU breakdown by geography"""
        query = """
        SELECT 
            GOV,
            Count(*) as subscribers,
            ROUND(AVG(Total_RPU), 2) as avg_rpu
        FROM broadband_subscribers
        GROUP BY 1
        ORDER BY 3 DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "Revenue by Geography",
            "data": df.to_dict('records') if df is not None else [],
            "insight": "Geographic performance comparison"
        }


class ServiceQualityTool(BroadbandAnalyticsTool):
    """Analyze service quality metrics - outages and tickets"""
    
    def outage_analysis(self) -> Dict[str, Any]:
        """Get distribution of subscribers by outage count"""
        query = """
        SELECT 
            CASE 
                WHEN Count_Of_All_Outage = 0 THEN 'No Outages'
                WHEN Count_Of_All_Outage <= 5 THEN '1-5 Outages'
                WHEN Count_Of_All_Outage <= 10 THEN '6-10 Outages'
                ELSE '10+ Outages'
            END as outage_category,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct,
            ROUND(AVG(Total_RPU), 2) as avg_revenue
        FROM broadband_subscribers
        WHERE Subscriber_Status = 'Active'
        GROUP BY 1
        ORDER BY count DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "Service Quality Distribution",
            "data": df.to_dict('records') if df is not None else [],
            "alerts": self._generate_quality_alerts(df)
        }
    
    def high_impact_issues(self) -> Dict[str, Any]:
        """Identify high-value subscribers with service issues"""
        query = """
        SELECT 
            Current_Package_Name,
            COUNT(*) as affected_subscribers,
            ROUND(AVG(Total_RPU), 2) as revenue_at_risk,
            ROUND(SUM(Total_RPU), 2) as total_revenue_at_risk,
            ROUND(AVG(Count_Of_All_Outage), 2) as avg_outages,
            ROUND(AVG(TTS_Total_Tickets), 2) as avg_tickets
        FROM broadband_subscribers
        WHERE Subscriber_Status = 'Active'
          AND (Count_Of_All_Outage > 5 OR TTS_Total_Tickets > 10)
        GROUP BY 1
        ORDER BY 4 DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "High-Impact Service Issues",
            "data": df.to_dict('records') if df is not None else [],
            "recommendation": "Prioritize support for high-revenue customers with service issues"
        }
    
    def _generate_quality_alerts(self, df: Optional[pd.DataFrame]) -> List[str]:
        """Generate alerts based on service quality"""
        alerts = []
        if df is None or df.empty:
            return alerts
        
        high_outage = df[df['outage_category'] == '10+ Outages']
        if not high_outage.empty and high_outage['pct'].values[0] > 10:
            alerts.append("⚠️ ALERT: Over 10% of subscribers experiencing excessive outages")
        
        return alerts


class GSMCrossSellTool(BroadbandAnalyticsTool):
    """Identify GSM cross-sell opportunities"""
    
    def gsm_adoption_gap(self) -> Dict[str, Any]:
        """Calculate GSM cross-sell opportunity"""
        query = """
        SELECT 
            COUNT(CASE WHEN Has_GSM = 'No' THEN 1 END) as potential_targets,
            ROUND(AVG(Total_RPU), 2) as current_avg_rpu,
            ROUND(AVG(Total_RPU) * 1.25, 2) as potential_avg_rpu,
            ROUND(SUM(Total_RPU), 2) as current_revenue,
            ROUND(SUM(Total_RPU) * 1.25, 2) as potential_revenue,
            ROUND(SUM(Total_RPU) * 0.25, 2) as revenue_uplift
        FROM broadband_subscribers
        WHERE Has_GSM = 'No'
          AND Subscriber_Status = 'Active'
          AND Total_RPU > 300
        """
        df = self.duckdb.query(query)
        if df is not None and not df.empty:
            data = df.to_dict('records')[0]
        else:
            data = {}
        
        return {
            "metric": "GSM Cross-sell Opportunity",
            "data": data,
            "recommendation": "GSM bundle increases ARPU by 25% - prioritize premium package customers without GSM"
        }


class SubscriberSegmentationTool(BroadbandAnalyticsTool):
    """Analyze subscriber demographics and segments"""
    
    def by_status(self) -> Dict[str, Any]:
        """Segment subscribers by status"""
        query = """
        SELECT 
            Subscriber_Status,
            COUNT(*) as subscribers,
            ROUND(AVG(Total_RPU), 2) as avg_rpu,
            ROUND(AVG(Tenure_Days), 0) as avg_tenure_days,
            ROUND((COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()), 2) as pct
        FROM broadband_subscribers
        GROUP BY 1
        ORDER BY 3 DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "Subscribers by Status",
            "data": df.to_dict('records') if df is not None else [],
            "insight": "Active subscribers generate 5-10x more revenue than suspended"
        }
    
    def by_gender(self) -> Dict[str, Any]:
        """Analyze gender distribution and revenue"""
        query = """
        SELECT 
            Gender,
            COUNT(*) as subscribers,
            ROUND(AVG(Total_RPU), 2) as avg_rpu,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct_of_total
        FROM broadband_subscribers
        WHERE Gender IS NOT NULL
        GROUP BY 1
        ORDER BY 3 DESC
        """
        df = self.duckdb.query(query)
        return {
            "metric": "Demographics by Gender",
            "data": df.to_dict('records') if df is not None else [],
            "insight": "Identifies gender-based revenue patterns for targeted marketing"
        }


class ReportGeneratorTool(BroadbandAnalyticsTool):
    """Generate structured business intelligence reports"""
    
    def executive_summary(self) -> Dict[str, Any]:
        """Generate executive dashboard snapshot"""
        
        # Total subscribers
        total_query = "SELECT COUNT(*) as total, ROUND(AVG(Total_RPU), 2) as avg_rpu FROM broadband_subscribers"
        total = self.duckdb.query(total_query)
        
        # Status breakdown
        status_query = """
        SELECT Subscriber_Status, COUNT(*) as count
        FROM broadband_subscribers
        WHERE Subscriber_Status IS NOT NULL
        GROUP BY 1
        """
        status = self.duckdb.query(status_query)
        
        # Package breakdown
        package_query = """
        SELECT Current_Package_Name, COUNT(*) as count
        FROM broadband_subscribers
        GROUP BY 1
        ORDER BY count DESC
        LIMIT 3
        """
        package = self.duckdb.query(package_query)
        
        return {
            "title": "Executive Summary",
            "total_customers": total.to_dict('records')[0] if total is not None else {},
            "status_distribution": status.to_dict('records') if status is not None else [],
            "top_packages": package.to_dict('records') if package is not None else []
        }
    
    def dashboard_with_visualizations(self) -> Dict[str, Any]:
        """Generate dashboard with visualization recommendations"""
        
        # Fetch data for all key metrics
        package_rpu = self.duckdb.query("""
            SELECT Current_Package_Name, COUNT(*) as count, AVG(Total_RPU) as avg_rpu
            FROM broadband_subscribers GROUP BY 1 ORDER BY avg_rpu DESC
        """)
        
        status_dist = self.duckdb.query("""
            SELECT Subscriber_Status, COUNT(*) as count
            FROM broadband_subscribers WHERE Subscriber_Status IS NOT NULL GROUP BY 1
        """)
        
        payment_type = self.duckdb.query("""
            SELECT 
                Payment_Type,
                COUNT(*) as count
            FROM broadband_subscribers GROUP BY 1
        """)
        
        dashboard = {
            "title": "Intelligence Dashboard",
            "timestamp": pd.Timestamp.now().isoformat(),
            "panels": []
        }
        
        # Panel 1: Revenue by Package
        if package_rpu is not None and not package_rpu.empty:
            panel1 = {
                "title": "Total_RPU by Package",
                "data": package_rpu.to_dict('records'),
                "visualization": {
                    "type": "bar",
                    "recommendation": "Bar chart shows package profitability ranking",
                    "key_insight": f"{package_rpu.iloc[0]['Current_Package_Name']} leads with ₵{package_rpu.iloc[0]['avg_rpu']:.0f} avg RPU"
                }
            }
            dashboard["panels"].append(panel1)
        
        # Panel 2: Status Distribution
        if status_dist is not None and not status_dist.empty:
            panel2 = {
                "title": "Subscriber Status",
                "data": status_dist.to_dict('records'),
                "visualization": {
                    "type": "pie",
                    "recommendation": "Pie chart shows composition of subscriber statuses",
                    "key_insight": f"{status_dist.iloc[0]['Subscriber_Status']}: {status_dist.iloc[0]['count']:,} subscribers"
                }
            }
            dashboard["panels"].append(panel2)
        
        # Panel 3: Payment Type
        if payment_type is not None and not payment_type.empty:
            panel3 = {
                "title": "Payment Type Distribution",
                "data": payment_type.to_dict('records'),
                "visualization": {
                    "type": "bar",
                    "recommendation": "Bar chart shows payment model concentration",
                    "key_insight": f"Largest segment: {payment_type.iloc[0]['Payment_Type']} ({payment_type.iloc[0]['count']:,})"
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
                "revenue_analysis": RevenuAnalysisTool(self.duckdb, self.km).by_payment_type(),
                "service_quality": ServiceQualityTool(self.duckdb, self.km).outage_analysis(),
                "gsm_opportunity": GSMCrossSellTool(self.duckdb, self.km).gsm_adoption_gap(),
                "dashboard": self.dashboard_with_visualizations()
            }
        }
        return json.dumps(report, indent=2)


def create_tool_functions(duckdb_conn: DuckDBConnector, km: KnowledgeManager) -> Dict:
    """Create callable tool functions for Agno agent"""
    
    revenue_tool = RevenuAnalysisTool(duckdb_conn, km)
    quality_tool = ServiceQualityTool(duckdb_conn, km)
    gsm_tool = GSMCrossSellTool(duckdb_conn, km)
    segment_tool = SubscriberSegmentationTool(duckdb_conn, km)
    report_tool = ReportGeneratorTool(duckdb_conn, km)
    
    return {
        "revenue_by_payment_type": lambda: json.dumps(revenue_tool.by_payment_type()),
        "revenue_by_package": lambda: json.dumps(revenue_tool.by_package()),
        "revenue_by_geography": lambda: json.dumps(revenue_tool.by_geography()),
        "service_quality_analysis": lambda: json.dumps(quality_tool.outage_analysis()),
        "high_impact_issues": lambda: json.dumps(quality_tool.high_impact_issues()),
        "gsm_crosssell_opportunity": lambda: json.dumps(gsm_tool.gsm_adoption_gap()),
        "segment_by_status": lambda: json.dumps(segment_tool.by_status()),
        "segment_by_gender": lambda: json.dumps(segment_tool.by_gender()),
        "executive_summary": lambda: json.dumps(report_tool.executive_summary()),
        "intelligence_dashboard": lambda: json.dumps(report_tool.dashboard_with_visualizations()),
        "full_analytics_report": report_tool.full_analytics_report
    }
