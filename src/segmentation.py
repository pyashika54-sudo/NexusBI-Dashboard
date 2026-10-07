import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set up logging for the segmentation module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def run_kmeans(df: pd.DataFrame, n_clusters: int = 3) -> Tuple[pd.DataFrame, List[str]]:
    """Selects numerical columns, standardizes them, and performs K-Means clustering.

    Args:
        df: The input pandas DataFrame.
        n_clusters: The number of clusters (default is 3).

    Returns:
        Tuple[pd.DataFrame, List[str]]: A tuple containing:
            - A copy of the DataFrame with a new 'cluster' column (values as categorical strings).
            - A list of the numerical columns used for clustering.
    """
    logger.info(f"Initiating K-Means clustering with n_clusters={n_clusters}.")
    
    if df.empty:
        logger.warning("Empty DataFrame passed to run_kmeans.")
        df_copy = df.copy()
        df_copy['cluster'] = pd.Series(dtype=str)
        return df_copy, []

    # 1. Select numerical columns
    numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numerical_cols:
        logger.error("No numerical columns found in DataFrame for K-Means clustering.")
        raise ValueError("The dataset must contain at least one numerical column for K-Means clustering.")
    
    logger.info(f"Selected numerical columns for clustering: {numerical_cols}")

    # Create a copy to avoid mutating the original df
    df_result = df.copy()

    # Drop NaNs or fill them (they should be filled by clean_data, but let's be safe)
    clustering_data = df_result[numerical_cols].copy()
    if clustering_data.isnull().any().any():
        logger.warning("NaN values found during clustering. Filling with 0.")
        clustering_data = clustering_data.fillna(0)

    # 2. Scale the numerical columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # 3. Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(scaled_data)

    # Add the cluster column as strings (categorical) for nicer Plotly coloring
    df_result['cluster'] = [f"Cluster {c}" for c in clusters]
    logger.info(f"Successfully computed {n_clusters} clusters.")
    
    return df_result, numerical_cols

def visualize_clusters(df: pd.DataFrame, numerical_cols: List[str], cluster_col: str = 'cluster') -> go.Figure:
    """Creates a 3D Plotly scatter plot (or 2D/1D if columns are limited) to visualize clusters.

    Args:
        df: The DataFrame containing the cluster assignments.
        numerical_cols: The list of numerical columns used during clustering.
        cluster_col: The column containing cluster labels.

    Returns:
        go.Figure: The Plotly figure object.
    """
    logger.info(f"Generating visualization for clusters using {len(numerical_cols)} dimensions.")
    
    if df.empty or cluster_col not in df.columns:
        logger.warning("DataFrame is empty or cluster column is missing. Returning empty figure.")
        return go.Figure()

    num_dims = len(numerical_cols)
    title = f"Customer Segments (K-Means Clustering)"

    try:
        # Custom gradient color palette
        custom_palette = ['#E94057', '#F27121', '#8A2387', '#3498db', '#2ecc71']

        # Fallback logic based on number of numerical columns available
        if num_dims >= 3:
            # 3D Scatter Plot. Map quantity, unitprice, and customerid specifically if present.
            quantity_col = next((c for c in numerical_cols if 'quantity' in c.lower()), None)
            unitprice_col = next((c for c in numerical_cols if 'unitprice' in c.lower() or 'price' in c.lower()), None)
            customer_col = next((c for c in numerical_cols if 'customerid' in c.lower() or 'customer' in c.lower() or 'id' in c.lower()), None)

            if quantity_col and unitprice_col and customer_col:
                x_col, y_col, z_col = quantity_col, unitprice_col, customer_col
            else:
                x_col, y_col, z_col = numerical_cols[0], numerical_cols[1], numerical_cols[2]

            fig = px.scatter_3d(
                df,
                x=x_col,
                y=y_col,
                z=z_col,
                color=cluster_col,
                title=title,
                labels={
                    x_col: x_col.replace("_", " ").title(),
                    y_col: y_col.replace("_", " ").title(),
                    z_col: z_col.replace("_", " ").title(),
                    cluster_col: "Segment"
                },
                color_discrete_sequence=custom_palette
            )
            fig.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)"),
                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)"),
                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)"),
                ),
                margin=dict(l=0, r=0, t=50, b=0)
            )
        elif num_dims == 2:
            # 2D Scatter Plot fallback
            x_col, y_col = numerical_cols[0], numerical_cols[1]
            logger.info("Only 2 numeric dimensions available. Falling back to 2D scatter plot.")
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=cluster_col,
                title=title + " (2D Fallback)",
                labels={
                    x_col: x_col.replace("_", " ").title(),
                    y_col: y_col.replace("_", " ").title(),
                    cluster_col: "Segment"
                },
                color_discrete_sequence=custom_palette
            )
        else:
            # 1D Strip Plot fallback
            x_col = numerical_cols[0]
            logger.info("Only 1 numeric dimension available. Falling back to 1D strip plot.")
            fig = px.strip(
                df,
                x=x_col,
                color=cluster_col,
                title=title + " (1D Fallback)",
                labels={
                    x_col: x_col.replace("_", " ").title(),
                    cluster_col: "Segment"
                },
                color_discrete_sequence=custom_palette
            )

        # Apply general premium layouts
        fig.update_layout(
            template="plotly_dark",
            title_font=dict(size=18, family="Outfit, sans-serif", color="#FFFFFF"),
            font=dict(family="Inter, sans-serif", color="#9CA3AF"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=600,
        )
        
        return fig

    except Exception as e:
        logger.error(f"Failed to generate cluster visualization: {str(e)}")
        raise
