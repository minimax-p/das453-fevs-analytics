import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="FEVS Analytics", layout="wide", page_icon="📊")


# Load data
@st.cache_data
def load_data():
    df_main = pd.read_excel('data/fevs_sample_data_3FYs_DataSet_1.xlsx', sheet_name='fevs_sample_data_3FYs_Set1')
    df_questions = pd.read_excel('data/fevs_sample_data_3FYs_DataSet_1.xlsx', sheet_name='Index-Qns-Map', skiprows=8)
    return df_main, df_questions


df_main, df_questions = load_data()


# Calculate percentages
@st.cache_data
def calc_perceptions(df_main):
    results = []
    question_cols = [col for col in df_main.columns if col.startswith('Q')]

    for q in question_cols:
        for year in [2023, 2024, 2025]:
            responses = df_main[df_main['FY'] == year][q].dropna()
            total = len(responses)
            if total > 0:
                pos = (responses >= 4).sum() / total * 100
                neu = (responses == 3).sum() / total * 100
                neg = (responses <= 2).sum() / total * 100
                results.append(
                    {'Question': q, 'Year': year, 'Positive': pos, 'Neutral': neu, 'Negative': neg, 'Total': total})
    return pd.DataFrame(results)


df_results = calc_perceptions(df_main)
pivot = df_results.pivot(index='Question', columns='Year', values='Positive')

# Identify patterns
strengths = pivot[pivot.min(axis=1) >= 70].reset_index()
weaknesses = pivot[pivot.max(axis=1) <= 40].reset_index()
improvement = pivot[(pivot.min(axis=1) > 40) & (pivot.min(axis=1) < 60)].reset_index()

# Add question text
q_map = df_questions[['Item.ID', 'Item.Text', 'Index', 'Sub.Index']].rename(columns={'Item.ID': 'Question'})
strengths = strengths.merge(q_map, on='Question', how='left')
improvement = improvement.merge(q_map, on='Question', how='left')
df_full = df_results.merge(q_map, on='Question', how='left')

# Title
st.title("📊 Federal Employee Viewpoint Survey Analytics")
st.markdown("### 3-Year Trend Analysis: 2023-2025")

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Responses", f"{len(df_main):,}")
col2.metric("Survey Questions", len(pivot))
col3.metric("Persistent Strengths (≥70%)", len(strengths))
col4.metric("Need Attention (<60%)", len(improvement) if len(weaknesses) == 0 else len(weaknesses))

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Executive Summary", "💪 Strengths", "📉 Areas for Improvement", "🔍 Question Drilldown"])

with tab1:
    st.header("Executive Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Findings")
        avg_positive = pivot.mean().mean()
        st.write(f"**Overall Positive Perception: {avg_positive:.1f}%**")

        if len(weaknesses) == 0:
            st.success("✓ No persistent weaknesses (≤40%) identified")

        pivot['Trend'] = pivot[2025] - pivot[2023]
        improving = len(pivot[pivot['Trend'] > 5])
        declining = len(pivot[pivot['Trend'] < -5])
        st.write(f"**Trends:** {improving} improving | {declining} declining")

    with col2:
        st.subheader("Performance by Index")
        index_avg = df_full.groupby('Index')['Positive'].mean().sort_values(ascending=False)
        fig = px.bar(index_avg, orientation='h', color=index_avg.values, color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="% Positive")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Biggest Changes (2023 → 2025)")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Most Improved**")
        top_improved = pivot.nlargest(5, 'Trend').reset_index()
        top_improved = top_improved.merge(q_map, on='Question')
        for _, row in top_improved.iterrows():
            st.write(f"**{row['Question']}** (+{row['Trend']:.1f}%): {row['Item.Text'][:60]}...")

    with col2:
        st.write("**Most Declined**")
        top_declined = pivot.nsmallest(5, 'Trend').reset_index()
        top_declined = top_declined.merge(q_map, on='Question')
        for _, row in top_declined.iterrows():
            st.write(f"**{row['Question']}** ({row['Trend']:.1f}%): {row['Item.Text'][:60]}...")

with tab2:
    st.header("💪 Persistent Strengths (≥70%)")

    strengths['Avg'] = strengths[[2023, 2024, 2025]].mean(axis=1)
    strengths = strengths.sort_values('Avg', ascending=False)

    top10 = strengths.head(10)
    fig = px.bar(top10, x='Avg', y='Item.Text', orientation='h', color='Avg', color_continuous_scale='Greens')
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="% Positive")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All Strengths")
    display_df = strengths[['Question', 'Item.Text', 'Index', 'Sub.Index', 2023, 2024, 2025, 'Avg']]
    st.dataframe(display_df, use_container_width=True, height=400)

with tab3:
    st.header("📉 Areas for Improvement")

    if len(weaknesses) > 0:
        st.subheader(f"⚠️ Persistent Weaknesses (≤40%): {len(weaknesses)}")
        st.dataframe(weaknesses)
    else:
        st.success("✓ No persistent weaknesses (≤40%) identified")

    st.subheader(f"📊 Lower Performing Questions (<60%): {len(improvement)}")

    improvement['Avg'] = improvement[[2023, 2024, 2025]].mean(axis=1)
    improvement = improvement.sort_values('Avg')

    bottom10 = improvement.head(10)
    fig = px.bar(bottom10, x='Avg', y='Item.Text', orientation='h', color='Avg', color_continuous_scale='Reds_r')
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="% Positive")
    st.plotly_chart(fig, use_container_width=True)

    display_df = improvement[['Question', 'Item.Text', 'Index', 'Sub.Index', 2023, 2024, 2025, 'Avg']]
    st.dataframe(display_df, use_container_width=True, height=400)

with tab4:
    st.header("🔍 Question-Level Drilldown")

    question_cols = sorted(pivot.index.tolist(), key=lambda x: int(x[1:]))
    selected_q = st.selectbox("Select a question:", question_cols)

    if selected_q:
        q_info = q_map[q_map['Question'] == selected_q].iloc[0]
        q_data = df_results[df_results['Question'] == selected_q]

        st.subheader(f"{selected_q}: {q_info['Item.Text']}")
        st.write(f"**Index:** {q_info['Index']} > {q_info['Sub.Index']}")

        col1, col2, col3 = st.columns(3)
        avg_pos = q_data['Positive'].mean()
        trend = pivot.loc[selected_q, 2025] - pivot.loc[selected_q, 2023]
        status = "Strength" if selected_q in strengths['Question'].values else "Needs Attention"

        col1.metric("Average Positive", f"{avg_pos:.1f}%")
        col2.metric("3-Year Change", f"{trend:+.1f}%")
        col3.metric("Status", status)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Positive'], mode='lines+markers',
                                 name='Positive %', line=dict(color='green', width=3)))
        fig.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Neutral'], mode='lines+markers',
                                 name='Neutral %', line=dict(color='gray', width=2)))
        fig.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Negative'], mode='lines+markers',
                                 name='Negative %', line=dict(color='red', width=2)))
        fig.update_layout(title=f"{selected_q} Trend", yaxis_title="Percentage", hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Yearly Breakdown")
        breakdown = q_data[['Year', 'Total', 'Positive', 'Neutral', 'Negative']].sort_values('Year')
        st.dataframe(breakdown, use_container_width=True)