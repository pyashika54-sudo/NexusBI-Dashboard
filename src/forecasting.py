import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Optional

# Set up logging for the forecasting module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def train_forecast_model(
    df: pd.DataFrame, 
    date_col: str, 
    target_col: str
) -> pd.DataFrame:
    """Aggregates the target variable by date, trains a forecasting model, and predicts the next 30 days.

    Attempts to use Prophet first. If Prophet is not installed or encounters an error, 
    falls back to a regression model using XGBoost.

    Args:
        df: The input pandas DataFrame.
        date_col: The column containing date information.
        target_col: The numerical column to forecast.

    Returns:
        pd.DataFrame: A DataFrame with the following structure:
            - ds: Date column (datetime)
            - actual: Historical aggregated values (NaN for future dates)
            - forecast: Predicted values (both historical fit and 30-day future predictions)
    """
    logger.info(f"Initiating forecast modeling for target '{target_col}' using date '{date_col}'.")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for forecasting.")
        return pd.DataFrame(columns=["ds", "actual", "forecast"])

    # 1. Prepare and aggregate historical data
    try:
        prep_df = df[[date_col, target_col]].copy()
        prep_df[date_col] = pd.to_datetime(prep_df[date_col])
        
        # Aggregate by date (sum is standard for sales/volumes; mean if needed. Let's do sum)
        history = prep_df.groupby(date_col)[target_col].sum().reset_index()
        history = history.sort_values(by=date_col).reset_index(drop=True)
        history.columns = ['ds', 'y']
        
        logger.info(f"Aggregated historical series has {len(history)} data points.")
        if len(history) < 2:
            raise ValueError("At least 2 historical data points are required for time-series forecasting.")
            
    except Exception as e:
        logger.error(f"Error prepping historical data for forecasting: {str(e)}")
        raise ValueError(f"Failed to prepare data: {str(e)}")

    # 2. Try Prophet forecasting
    try:
        logger.info("Attempting Prophet forecast model.")
        from prophet import Prophet
        
        # Suppress Prophet verbose logs
        import os
        logging.getLogger('prophet').setLevel(logging.ERROR)
        
        # Initialize and fit
        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
        model.fit(history)
        
        # Generate 30-day future periods
        future = model.make_future_dataframe(periods=30, freq='D')
        forecast_res = model.predict(future)
        
        # Merge actual and forecast
        result = pd.merge(forecast_res[['ds', 'yhat']], history, on='ds', how='left')
        result.columns = ['ds', 'forecast', 'actual']
        
        logger.info("Prophet forecast model completed successfully.")
        return result[['ds', 'actual', 'forecast']]
        
    except Exception as e_prophet:
        logger.warning(f"Prophet failed or is unavailable (Error: {str(e_prophet)}). Falling back to XGBoost.")
        
        # 3. XGBoost fallback forecaster
        try:
            from xgboost import XGBRegressor
            
            # Feature engineering for time series
            history['trend'] = np.arange(len(history))
            history['month'] = history['ds'].dt.month
            history['day'] = history['ds'].dt.day
            history['dayofweek'] = history['ds'].dt.dayofweek
            
            X_train = history[['trend', 'month', 'day', 'dayofweek']]
            y_train = history['y']
            
            # Train model
            xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
            xgb_model.fit(X_train, y_train)
            
            # Create future dates (next 30 days)
            last_date = history['ds'].max()
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
            
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Combine history and future for trend indexing and feature generation
            combined = pd.concat([history[['ds']], future_df], ignore_index=True)
            combined['trend'] = np.arange(len(combined))
            combined['month'] = combined['ds'].dt.month
            combined['day'] = combined['ds'].dt.day
            combined['dayofweek'] = combined['ds'].dt.dayofweek
            
            # Run prediction across entire series
            X_pred = combined[['trend', 'month', 'day', 'dayofweek']]
            predictions = xgb_model.predict(X_pred)
            
            # Merge back into structured output
            combined['forecast'] = predictions
            combined = pd.merge(combined, history[['ds', 'y']], on='ds', how='left')
            combined.rename(columns={'y': 'actual'}, inplace=True)
            
            logger.info("XGBoost fallback forecaster completed successfully.")
            return combined[['ds', 'actual', 'forecast']]
            
        except Exception as e_xgb:
            logger.warning(f"XGBoost failed (Error: {str(e_xgb)}). Falling back to Linear Trend Regression.")
            
            # 4. Simple Linear Trend Regression fallback (No dependencies failed)
            from sklearn.linear_model import LinearRegression
            
            history['trend'] = np.arange(len(history))
            X_train = history[['trend']]
            y_train = history['y']
            
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            
            last_date = history['ds'].max()
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
            future_df = pd.DataFrame({'ds': future_dates})
            
            combined = pd.concat([history[['ds']], future_df], ignore_index=True)
            combined['trend'] = np.arange(len(combined))
            
            predictions = lr_model.predict(combined[['trend']])
            
            combined['forecast'] = predictions
            combined = pd.merge(combined, history[['ds', 'y']], on='ds', how='left')
            combined.rename(columns={'y': 'actual'}, inplace=True)
            
            logger.info("Linear Trend Regression fallback completed successfully.")
            return combined[['ds', 'actual', 'forecast']]

def visualize_forecast(forecast_df: pd.DataFrame, target_name: str = "Metric") -> go.Figure:
    """Generates a premium Plotly line chart representing actual and forecasted target metrics.

    Plots the historical actual series in blue and the predicted (fitted + future) values in orange.

    Args:
        forecast_df: The DataFrame output from train_forecast_model.
        target_name: Name of the variable being forecasted (for axis titles).

    Returns:
        go.Figure: The Plotly figure object.
    """
    logger.info("Generating line chart for forecast visualization.")
    
    if forecast_df.empty:
        logger.warning("Empty forecast DataFrame. Returning empty figure.")
        return go.Figure()

    try:
        fig = go.Figure()

        # Trace 1: Historical Actuals (Gradient theme main color)
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['actual'],
            name='Historical Actual',
            mode='lines',
            line=dict(color='#FF3366', width=3),
            connectgaps=False
        ))

        # Trace 2: Predicted Future (Gradient theme secondary color)
        # We split predictions to show future 30 days distinctly, but connecting smoothly
        # Find where historical data ends
        historical_len = forecast_df['actual'].notna().sum()
        
        # Future predictions start at index (historical_len - 1) to connect visual lines
        future_part = forecast_df.iloc[max(0, historical_len - 1):]
        
        fig.add_trace(go.Scatter(
            x=future_part['ds'],
            y=future_part['forecast'],
            name='30-Day Forecast',
            mode='lines',
            line=dict(color='#20D2EB', width=3, dash='dash')
        ))

        # Apply custom design/layouts
        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            title=f"30-Day Forecast Projection: {target_name.replace('_', ' ').title()}",
            title_font=dict(size=18, family="Outfit, sans-serif", color="#FFFFFF"),
            font=dict(family="Inter, sans-serif", color="#9CA3AF"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", linecolor="#CBD5E1"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", linecolor="#CBD5E1"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        return fig

    except Exception as e:
        logger.error(f"Failed to generate forecast line chart: {str(e)}")
        raise
