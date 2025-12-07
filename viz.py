import plotly.express as px
import pandas as pd

def plot_df_sample(df: pd.DataFrame, kind: str, x: str, y: str):
    # simple wrapper to create a plotly figure
    if kind == "line":
        fig = px.line(df, x=x, y=y, title=f"{y} over {x}")
    elif kind == "hist" or kind == "histogram":
        fig = px.histogram(df, x=x, title=f"Distribution of {x}")
    else:
        fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
    return fig
