import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Set up logging for the EDA engine
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def generate_bar_chart(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: Optional[str] = None, 
    title: Optional[str] = None,
    color_discrete_sequence: Optional[list] = None
) -> go.Figure:
    """Generates a premium, styled Plotly bar chart for categorical distributions or relationships.

    If y_col is not specified, it will count the occurrences of categories in x_col.

    Args:
        df: The pandas DataFrame containing the data.
        x_col: The column name to use for the X-axis (categories).
        y_col: Optional column name for the Y-axis (values). If None, frequency counts are shown.
        title: Title of the chart.
        color_discrete_sequence: Optional custom color palette list.

    Returns:
        go.Figure: A Plotly graph object figure.
    """
    logger.info(f"Generating bar chart for column '{x_col}' (y_col: '{y_col}').")
    
    if df.empty:
        logger.warning("Empty DataFrame passed to generate_bar_chart.")
        return go.Figure()

    if x_col not in df.columns:
        logger.error(f"X-axis column '{x_col}' not found in DataFrame.")
        raise ValueError(f"Column '{x_col}' does not exist in the DataFrame.")

    # Cast X column to string before generating charts
    df[x_col] = df[x_col].astype(str)

    try:
        if y_col is None:
            # Aggregate counts if no Y-axis values are provided
            counts_df = df[x_col].value_counts().reset_index()
            counts_df.columns = [x_col, "count"]
            fig = px.bar(
                counts_df,
                x=x_col,
                y="count",
                title=title or f"Distribution of {x_col.replace('_', ' ').title()}",
                color="count",
                color_continuous_scale=px.colors.sequential.Plasma,
                labels={x_col: x_col.replace("_", " ").title(), "count": "Count"}
            )
        else:
            if y_col not in df.columns:
                logger.error(f"Y-axis column '{y_col}' not found in DataFrame.")
                raise ValueError(f"Column '{y_col}' does not exist in the DataFrame.")
                
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                title=title or f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                color=y_col,
                color_continuous_scale=px.colors.sequential.Plasma,
                labels={
                    x_col: x_col.replace("_", " ").title(),
                    y_col: y_col.replace("_", " ").title()
                }
            )

        # Apply custom premium styling layout
        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            title_font=dict(size=18, family="Outfit, sans-serif", color="#FFFFFF"),
            font=dict(family="Inter, sans-serif", color="#9CA3AF"),
            xaxis=dict(showgrid=False, linecolor="#CBD5E1"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", linecolor="#CBD5E1"),
            margin=dict(l=40, r=40, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    except Exception as e:
        logger.error(f"Failed to generate bar chart: {str(e)}")
        raise

def generate_line_chart(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    title: Optional[str] = None,
    color: Optional[str] = None
) -> go.Figure:
    """Generates a premium, styled Plotly line chart for numerical trends.

    Args:
        df: The pandas DataFrame containing the data.
        x_col: The column name for the X-axis (e.g. dates, trend indicators).
        y_col: The column name for the Y-axis (numerical value).
        title: Title of the chart.
        color: Hex color string for the line.

    Returns:
        go.Figure: A Plotly graph object figure.
    """
    logger.info(f"Generating line chart for '{x_col}' vs '{y_col}'.")
    
    if df.empty:
        logger.warning("Empty DataFrame passed to generate_line_chart.")
        return go.Figure()

    if x_col not in df.columns or y_col not in df.columns:
        logger.error(f"Required columns '{x_col}' or '{y_col}' not found in DataFrame.")
        raise ValueError(f"Columns '{x_col}' or '{y_col}' must exist in the DataFrame.")

    # Cast X column to string before generating charts
    df[x_col] = df[x_col].astype(str)

    # Use a modern default line color if none specified (high-contrast neon line)
    line_color = color or "#00FF7F"

    try:
        # Sort by the X-axis to ensure line renders sequentially
        sorted_df = df.sort_values(by=x_col)

        fig = px.line(
            sorted_df,
            x=x_col,
            y=y_col,
            title=title or f"{y_col.replace('_', ' ').title()} trend over {x_col.replace('_', ' ').title()}",
            labels={
                x_col: x_col.replace("_", " ").title(),
                y_col: y_col.replace("_", " ").title()
            }
        )

        # Style the line specifically (thick neon line)
        fig.update_traces(line=dict(color=line_color, width=4), mode="lines+markers")

        # Apply custom layout styles
        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            title_font=dict(size=18, family="Outfit, sans-serif", color="#FFFFFF"),
            font=dict(family="Inter, sans-serif", color="#9CA3AF"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", linecolor="#CBD5E1"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", linecolor="#CBD5E1"),
            margin=dict(l=40, r=40, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    except Exception as e:
        logger.error(f"Failed to generate line chart: {str(e)}")
        raise
