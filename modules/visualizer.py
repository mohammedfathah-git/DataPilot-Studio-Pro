"""
DataPilot Studio Pro - Executive Visualization Studio
Provides high-impact Plotly visualizations with responsive executive styling,
color themes, data labels, and light/dark mode font contrast.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

THEME_PALETTES = {
    "Indigo Modern": ["#6366F1", "#8B5CF6", "#EC4899", "#3B82F6", "#10B981", "#F59E0B"],
    "Cyberpunk Emerald": ["#10B981", "#06B6D4", "#8B5CF6", "#F43F5E", "#EAB308", "#3B82F6"],
    "Sunset Warm": ["#F43F5E", "#FB923C", "#FBBF24", "#A855F7", "#6366F1", "#14B8A6"],
    "Corporate Navy": ["#1E3A8A", "#2563EB", "#60A5FA", "#93C5FD", "#0D9488", "#64748B"]
}

def style_figure(fig, is_dark: bool = True, title: str = ""):
    """Applies executive, modern typography and clean backgrounds to Plotly figures."""
    font_color = "#F1F5F9" if is_dark else "#0F172A"
    tick_color = "#CBD5E1" if is_dark else "#334155"
    grid_color = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.07)"
    bg_color = "rgba(0,0,0,0)"
    
    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(size=16, color=font_color, family="Plus Jakarta Sans, sans-serif"),
            x=0.01,
            y=0.96
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=font_color, size=12),
        margin=dict(l=25, r=25, t=55 if title else 25, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=font_color)
        ),
        hoverlabel=dict(
            bgcolor="#1E1B4B" if is_dark else "#FFFFFF",
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            font_color="#FFFFFF" if is_dark else "#0F172A"
        )
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        linecolor=tick_color,
        tickfont=dict(color=tick_color, size=11, family="Plus Jakarta Sans, sans-serif"),
        title_font=dict(color=font_color, size=13, family="Plus Jakarta Sans, sans-serif")
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        linecolor=tick_color,
        tickfont=dict(color=tick_color, size=11, family="Plus Jakarta Sans, sans-serif"),
        title_font=dict(color=font_color, size=13, family="Plus Jakarta Sans, sans-serif")
    )
    return fig

def render_chart_with_actions(fig, key: str, is_dark: bool = True, title: str = "", filename: str = "chart.html"):
    """Renders Plotly chart with direct standalone interactive HTML export."""
    styled_fig = style_figure(fig, is_dark=is_dark, title=title)
    st.plotly_chart(styled_fig, use_container_width=True, key=key)
    return styled_fig

def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str = None,
    barmode: str = "group",
    orientation: str = "v",
    palette: str = "Indigo Modern",
    show_labels: bool = True
) -> go.Figure:
    colors = THEME_PALETTES.get(palette, THEME_PALETTES["Indigo Modern"])
    
    if orientation == "h":
        fig = px.bar(
            df, x=y, y=x, color=color, barmode=barmode, orientation="h",
            color_discrete_sequence=colors,
            text_auto=True if show_labels else False
        )
    else:
        fig = px.bar(
            df, x=x, y=y, color=color, barmode=barmode, orientation="v",
            color_discrete_sequence=colors,
            text_auto=True if show_labels else False
        )
        
    if show_labels:
        fig.update_traces(textposition="outside", cliponaxis=False)
    return fig

def create_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str = None,
    markers: bool = True,
    palette: str = "Indigo Modern",
    show_labels: bool = True
) -> go.Figure:
    colors = THEME_PALETTES.get(palette, THEME_PALETTES["Indigo Modern"])
    
    if show_labels:
        # Round or format labels nicely for lines
        fig = px.line(
            df, x=x, y=y, color=color, markers=markers,
            text=y,
            color_discrete_sequence=colors
        )
        fig.update_traces(textposition="top center")
    else:
        fig = px.line(df, x=x, y=y, color=color, markers=markers, color_discrete_sequence=colors)
    return fig

def create_scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: str = None,
    color: str = None,
    trendline: bool = False,
    palette: str = "Indigo Modern",
    show_labels: bool = False
) -> go.Figure:
    colors = THEME_PALETTES.get(palette, THEME_PALETTES["Indigo Modern"])
    fig = px.scatter(
        df, x=x, y=y, size=size, color=color,
        trendline="ols" if trendline and pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]) else None,
        text=df.index if show_labels else None,
        color_discrete_sequence=colors
    )
    if show_labels:
        fig.update_traces(textposition="top center")
    return fig


def create_distribution_chart(
    df: pd.DataFrame,
    col: str,
    kind: str = "Histogram",
    nbins: int = 30,
    color: str = None,
    palette: str = "Indigo Modern",
    show_labels: bool = True
) -> go.Figure:

    colors = THEME_PALETTES.get(
        palette,
        THEME_PALETTES["Indigo Modern"]
    )

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")

    if kind == "Histogram":
        plot_df = df.copy()

        # Convert histogram values to numeric
        plot_df[col] = pd.to_numeric(
            plot_df[col], errors="coerce"
        )
        plot_df = plot_df.dropna(subset=[col])

        if plot_df.empty:
            raise ValueError(
                f"No valid numeric values available for {col}."
            )

        fig = px.histogram(
            plot_df,
            x=col,
            nbins=nbins,
            color=color if color in plot_df.columns else None,
            marginal="box",
            color_discrete_sequence=colors
        )

        if show_labels:
            fig.update_traces(
                hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>"
            )

    elif kind == "Box Plot":
        fig = px.box(
            df,
            y=col,
            x=color,
            color=color,
            color_discrete_sequence=colors
        )

    elif kind == "Violin Plot":
        fig = px.violin(
            df,
            y=col,
            x=color,
            color=color,
            box=True,
            points="all",
            color_discrete_sequence=colors
        )

    else:
        raise ValueError(f"Unsupported distribution chart: {kind}")

    return fig


def create_composition_chart(
    df: pd.DataFrame,
    names: str,
    values: str,
    kind: str = "Donut",
    palette: str = "Indigo Modern",
    show_labels: bool = True
) -> go.Figure:
    colors = THEME_PALETTES.get(palette, THEME_PALETTES["Indigo Modern"])
    if kind == "Donut":
        fig = px.pie(df, names=names, values=values, hole=0.45, color_discrete_sequence=colors)
        if show_labels:
            fig.update_traces(textinfo="label+value+percent", textposition="inside")
    elif kind == "Pie":
        fig = px.pie(df, names=names, values=values, color_discrete_sequence=colors)
        if show_labels:
            fig.update_traces(textinfo="label+value+percent", textposition="inside")
    elif kind == "Treemap":
        fig = px.treemap(df, path=[names], values=values, color_discrete_sequence=colors)
    return fig

def create_heatmap_chart(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Bivariate Correlation Heatmap"
    )
    return fig
