import plotly.express as px
import plotly.graph_objects as go


PALETTE = px.colors.qualitative.Set2


def risk_distribution_donut(df, risk_col="health_condition"):
    counts = df[risk_col].value_counts().reset_index()
    counts.columns = [risk_col, "count"]
    fig = px.pie(
        counts, names=risk_col, values="count", hole=0.5,
        title="Health Risk Distribution", color_discrete_sequence=PALETTE,
    )
    fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value} students (%{percent})")
    return fig


def histogram(df, col, title=None, nbins=30):
    fig = px.histogram(
        df, x=col, nbins=nbins, marginal="box",
        title=title or f"Distribution of {col.replace('_', ' ').title()}",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(xaxis_title=col.replace("_", " ").title(), yaxis_title="Number of Students")
    return fig


def category_bar(df, col, title=None):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(
        counts, x=col, y="count", title=title or f"{col.replace('_', ' ').title()} Distribution",
        color=col, color_discrete_sequence=PALETTE, text="count",
    )
    fig.update_layout(xaxis_title=col.replace("_", " ").title(), yaxis_title="Number of Students", showlegend=False)
    return fig


def scatter(df, x, y, color=None, title=None):
    fig = px.scatter(
        df, x=x, y=y, color=color, opacity=0.6,
        title=title or f"{x.replace('_', ' ').title()} vs {y.replace('_', ' ').title()}",
        color_discrete_sequence=PALETTE, trendline=None,
    )
    fig.update_layout(xaxis_title=x.replace("_", " ").title(), yaxis_title=y.replace("_", " ").title())
    return fig


def box_plot(df, x, y, title=None):
    fig = px.box(
        df, x=x, y=y, color=x,
        title=title or f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(xaxis_title=x.replace("_", " ").title(), yaxis_title=y.replace("_", " ").title(), showlegend=False)
    return fig


def correlation_heatmap(df, cols, title="Correlation Heatmap"):
    corr = df[cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto", title=title,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    return fig


def grouped_bar(df, x, color, title=None):
    counts = df.groupby([x, color]).size().reset_index(name="count")
    fig = px.bar(
        counts, x=x, y="count", color=color, barmode="group",
        title=title or f"{color.replace('_', ' ').title()} by {x.replace('_', ' ').title()}",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(xaxis_title=x.replace("_", " ").title(), yaxis_title="Number of Students")
    return fig


def gauge_chart(score: int, title="Student Health Score"):
    if score >= 75:
        bar_color = "#2ecc71"
    elif score >= 50:
        bar_color = "#f1c40f"
    else:
        bar_color = "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 50], "color": "#fdecea"},
                {"range": [50, 75], "color": "#fff8e1"},
                {"range": [75, 100], "color": "#e8f8f0"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig
