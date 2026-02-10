"""
Visualization Module - Generate interactive charts with Plotly
Integrates with Agno agent to provide visual + textual insights
"""
import json
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types"""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    BOX = "box"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    FUNNEL = "funnel"
    GAUGE = "gauge"
    TABLE = "table"


class ChartDetector:
    """Detect optimal chart type from question and data"""
    
    keyword_map = {
        # Distribution charts
        'distribution': ChartType.PIE,
        'share': ChartType.PIE,
        'composition': ChartType.PIE,
        'breakdown': ChartType.BAR,
        
        # Comparison charts
        'compare': ChartType.BAR,
        'comparison': ChartType.BAR,
        'versus': ChartType.BAR,
        'vs': ChartType.BAR,
        'top': ChartType.BAR,
        'best': ChartType.BAR,
        'worst': ChartType.BAR,
        'ranking': ChartType.BAR,
        
        # Trend charts
        'trend': ChartType.LINE,
        'over time': ChartType.LINE,
        'growth': ChartType.LINE,
        'decline': ChartType.LINE,
        'progress': ChartType.LINE,
        'forecast': ChartType.LINE,
        
        # Relationship charts
        'correlation': ChartType.SCATTER,
        'relationship': ChartType.SCATTER,
        'scatter': ChartType.SCATTER,
        'outlier': ChartType.BOX,
        'distribution by': ChartType.BOX,
        
        # Risk/Performance charts
        'risk': ChartType.GAUGE,
        'performance': ChartType.GAUGE,
        'score': ChartType.GAUGE,
    }
    
    @staticmethod
    def detect(question: str, df: pd.DataFrame) -> ChartType:
        """Detect best chart type for question"""
        question_lower = question.lower()
        
        # Check keywords
        for keyword, chart_type in ChartDetector.keyword_map.items():
            if keyword in question_lower:
                return chart_type
        
        # Analyze data structure
        num_cols = len(df.select_dtypes(include=['number']).columns)
        text_cols = len(df.select_dtypes(include=['object']).columns)
        rows = len(df)
        
        # Data-driven decision
        if rows > 100 and num_cols >= 2:
            return ChartType.SCATTER
        elif text_cols >= 1 and num_cols >= 1:
            return ChartType.BAR
        else:
            return ChartType.BAR
    
    @staticmethod
    def suggest_multiple(question: str, df: pd.DataFrame) -> List[ChartType]:
        """Suggest multiple chart types"""
        primary = ChartDetector.detect(question, df)
        suggestions = [primary]
        
        # Add complementary charts
        if primary == ChartType.BAR:
            suggestions.append(ChartType.PIE)
        elif primary == ChartType.LINE:
            suggestions.append(ChartType.BAR)
        elif primary == ChartType.SCATTER:
            suggestions.append(ChartType.HEATMAP)
        
        return suggestions


class ChartBuilder:
    """Build interactive Plotly charts from data"""
    
    @staticmethod
    def build(df: pd.DataFrame, chart_type: ChartType, 
              title: str = "", x_col: Optional[str] = None, 
              y_col: Optional[str] = None) -> Optional[go.Figure]:
        """Build chart from DataFrame"""
        
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to ChartBuilder")
            return None
        
        try:
            if chart_type == ChartType.BAR:
                return ChartBuilder._build_bar(df, title, x_col, y_col)
            elif chart_type == ChartType.LINE:
                return ChartBuilder._build_line(df, title, x_col, y_col)
            elif chart_type == ChartType.PIE:
                return ChartBuilder._build_pie(df, title, x_col, y_col)
            elif chart_type == ChartType.SCATTER:
                return ChartBuilder._build_scatter(df, title, x_col, y_col)
            elif chart_type == ChartType.BOX:
                return ChartBuilder._build_box(df, title, x_col, y_col)
            elif chart_type == ChartType.HISTOGRAM:
                return ChartBuilder._build_histogram(df, title, y_col)
            elif chart_type == ChartType.GAUGE:
                return ChartBuilder._build_gauge(df, title)
            elif chart_type == ChartType.TABLE:
                return ChartBuilder._build_table(df, title)
            else:
                return ChartBuilder._build_bar(df, title, x_col, y_col)
        except Exception as e:
            logger.error(f"Chart building failed: {str(e)}")
            return None
    
    @staticmethod
    def _build_bar(df: pd.DataFrame, title: str, x_col: Optional[str], 
                   y_col: Optional[str]) -> go.Figure:
        """Build bar chart"""
        if x_col is None or y_col is None:
            x_col, y_col = ChartBuilder._auto_detect_columns(df)
        
        # Sort by y value
        try:
            df_sorted = df.sort_values(y_col, ascending=False).head(20)
        except:
            df_sorted = df.head(20)
        
        fig = px.bar(df_sorted, x=x_col, y=y_col, 
                     title=title or f"{y_col} by {x_col}",
                     text=y_col, color=y_col,
                     color_continuous_scale='Blues')
        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig.update_layout(
            height=400,
            showlegend=False,
            hovermode='x unified'
        )
        return fig
    
    @staticmethod
    def _build_line(df: pd.DataFrame, title: str, x_col: Optional[str],
                    y_col: Optional[str]) -> go.Figure:
        """Build line chart"""
        if x_col is None or y_col is None:
            # Find date column or use index
            x_col = ChartBuilder._find_date_column(df) or df.columns[0]
            y_col = [col for col in df.columns if col != x_col 
                    and df[col].dtype in ['float64', 'int64']][0] if len(df.columns) > 1 else df.columns[1]
        
        try:
            df_sorted = df.sort_values(x_col)
        except:
            df_sorted = df
        
        fig = px.line(df_sorted, x=x_col, y=y_col,
                     title=title or f"{y_col} Trend",
                     markers=True)
        fig.update_layout(
            height=400,
            hovermode='x unified'
        )
        return fig
    
    @staticmethod
    def _build_pie(df: pd.DataFrame, title: str, x_col: Optional[str],
                   y_col: Optional[str]) -> go.Figure:
        """Build pie chart"""
        if x_col is None or y_col is None:
            x_col, y_col = ChartBuilder._auto_detect_columns(df)
        
        fig = px.pie(df.head(15), names=x_col, values=y_col,
                    title=title or f"{x_col} Distribution",
                    hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def _build_scatter(df: pd.DataFrame, title: str, x_col: Optional[str],
                      y_col: Optional[str]) -> go.Figure:
        """Build scatter chart"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return ChartBuilder._build_bar(df, title, x_col, y_col)
        
        x_col = x_col or numeric_cols[0]
        y_col = y_col or numeric_cols[1]
        
        fig = px.scatter(df, x=x_col, y=y_col,
                        title=title or f"{y_col} vs {x_col}",
                        size=y_col if len(df) < 100 else None,
                        hover_data=df.columns.tolist()[:5])
        fig.update_layout(height=400, hovermode='closest')
        return fig
    
    @staticmethod
    def _build_box(df: pd.DataFrame, title: str, x_col: Optional[str],
                   y_col: Optional[str]) -> go.Figure:
        """Build box plot"""
        if x_col is None or y_col is None:
            x_col, y_col = ChartBuilder._auto_detect_columns(df)
        
        fig = px.box(df, x=x_col, y=y_col,
                    title=title or f"{y_col} Distribution")
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def _build_histogram(df: pd.DataFrame, title: str, 
                        y_col: Optional[str]) -> go.Figure:
        """Build histogram"""
        if y_col is None:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            y_col = numeric_cols[0] if numeric_cols else df.columns[0]
        
        fig = px.histogram(df, x=y_col,
                          title=title or f"{y_col} Distribution",
                          nbins=30)
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def _build_gauge(df: pd.DataFrame, title: str) -> go.Figure:
        """Build gauge chart (for KPIs)"""
        # Try to extract a single metric
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            value = df[numeric_cols[0]].iloc[0] if len(df) > 0 else 0
            value = min(100, max(0, float(value)))  # Normalize to 0-100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                title={'text': title or "KPI"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ]
                }
            ))
            fig.update_layout(height=400)
            return fig
        
        return ChartBuilder._build_bar(df, title, None, None)
    
    @staticmethod
    def _build_table(df: pd.DataFrame, title: str) -> go.Figure:
        """Build table visualization"""
        fig = go.Figure(data=[go.Table(
            header=dict(values=df.columns,
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[df[col] for col in df.columns],
                      fill_color='lavender',
                      align='left')
        )])
        fig.update_layout(height=400, title=title or "Data Table")
        return fig
    
    @staticmethod
    def _auto_detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
        """Auto-detect x and y columns"""
        # Prefer string/object for x, numeric for y
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        x_col = text_cols[0] if text_cols else df.columns[0]
        y_col = numeric_cols[0] if numeric_cols else df.columns[-1]
        
        return x_col, y_col
    
    @staticmethod
    def _find_date_column(df: pd.DataFrame) -> Optional[str]:
        """Find date/time column"""
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower():
                return col
        return None


class VisualizationManager:
    """Manage visualization suggestions and generation"""
    
    def __init__(self):
        self.chart_detector = ChartDetector()
        self.chart_builder = ChartBuilder()
    
    def suggest_visualizations(self, question: str, df: pd.DataFrame, 
                              data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest visualizations for question and data"""
        
        if df is None or df.empty:
            return {"charts": [], "recommendation": "No data to visualize"}
        
        # Detect chart types
        primary_chart = self.chart_detector.detect(question, df)
        suggested_charts = self.chart_detector.suggest_multiple(question, df)
        
        # Generate charts
        charts = []
        for i, chart_type in enumerate(suggested_charts[:2]):  # Limit to 2 charts
            fig = self.chart_builder.build(df, chart_type, 
                                          title=f"Analysis {i+1}")
            if fig:
                charts.append({
                    "type": chart_type.value,
                    "figure": fig,
                    "description": self._describe_chart(chart_type)
                })
        
        return {
            "primary_chart": primary_chart.value,
            "charts": charts,
            "recommendation": self._recommend_visualization(question, df),
            "data": df.to_dict('records')[:10]  # First 10 rows for context
        }
    
    def _describe_chart(self, chart_type: ChartType) -> str:
        """Describe what the chart shows"""
        descriptions = {
            ChartType.BAR: "Compare values across categories",
            ChartType.LINE: "Show trends over time",
            ChartType.PIE: "Display composition and parts of whole",
            ChartType.SCATTER: "Show relationship between variables",
            ChartType.BOX: "Analyze distribution and outliers",
            ChartType.HISTOGRAM: "View data distribution",
            ChartType.GAUGE: "Monitor KPI performance",
            ChartType.TABLE: "View detailed data"
        }
        return descriptions.get(chart_type, "Visualize data")
    
    def _recommend_visualization(self, question: str, df: pd.DataFrame) -> str:
        """Provide visualization recommendation"""
        primary = self.chart_detector.detect(question, df)
        
        recommendations = {
            ChartType.BAR: "Bar chart best shows rankings and comparisons",
            ChartType.LINE: "Line chart reveals trends and patterns over time",
            ChartType.PIE: "Pie chart illustrates market share and distribution",
            ChartType.SCATTER: "Scatter plot highlights correlations",
            ChartType.GAUGE: "Gauge tracks performance against targets",
        }
        
        return recommendations.get(primary, "Visual analysis ready")


def create_visualization_metadata(question: str, df: pd.DataFrame,
                                  analysis: Dict[str, Any]) -> str:
    """Create visualization metadata as JSON for agent"""
    
    manager = VisualizationManager()
    viz_suggestions = manager.suggest_visualizations(question, df, analysis)
    
    # Convert Plotly figures to JSON for transmission
    chart_specs = []
    for chart in viz_suggestions.get('charts', []):
        chart_specs.append({
            "type": chart['type'],
            "title": chart.get('title', ''),
            "description": chart['description']
        })
    
    metadata = {
        "visualization_recommended": len(chart_specs) > 0,
        "primary_chart_type": viz_suggestions.get('primary_chart'),
        "charts": chart_specs,
        "recommendation": viz_suggestions.get('recommendation'),
        "data_rows": len(df),
        "data_preview": df.head(5).to_dict('records')
    }
    
    return json.dumps(metadata)
