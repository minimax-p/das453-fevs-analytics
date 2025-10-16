import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Reduce

st.set_page_config(layout="wide", page_title="FEVS Analytics", page_icon="📊")

# Load data
@st.cache_data
def load_data():
    df_main = pd.read_excel('data/fevs_sample_data_3FYs_DataSet_1.xlsx', sheet_name='fevs_sample_data_3FYs_Set1')
    df_questions = pd.read_excel('data/fevs_sample_data_3FYs_DataSet_1.xlsx', sheet_name='Index-Qns-Map', skiprows=8)
    df_indices = pd.read_excel('data/fevs_sample_data_3FYs_DataSet_1.xlsx', sheet_name='Index-Def-Map')
    return df_main, df_questions, df_indices

df_main, df_questions, df_indices = load_data()

# Calculate percentages
@st.cache_data
def calc_data(df_main):
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
                results.append({'Question': q, 'Year': year, 'Positive': pos, 'Neutral': neu, 'Negative': neg})
    return pd.DataFrame(results)

df_results = calc_data(df_main)
pivot = df_results.pivot(index='Question', columns='Year', values='Positive')
q_map = df_questions[['Item.ID', 'Item.Text', 'Index', 'Sub.Index']].rename(columns={'Item.ID': 'Question'})
df_full = df_results.merge(q_map, on='Question')

# Sidebar
st.sidebar.title("🔧 Dashboard Controls")
st.sidebar.markdown("---")

selected_index = st.sidebar.selectbox("Filter by Index:", ['All'] + sorted(df_full['Index'].dropna().unique().tolist()))
strength_threshold = st.sidebar.slider("Strength Threshold (%):", 60, 90, 70, 5)
weakness_threshold = st.sidebar.slider("Improvement Threshold (%):", 30, 60, 60, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("Question Drilldown")
question_cols = sorted(pivot.index.tolist(), key=lambda x: int(x[1:]))
selected_question = st.sidebar.selectbox("Select Question:", ['None'] + question_cols)

st.sidebar.markdown("---")
if st.sidebar.button("📥 Download Full Results"):
    csv = df_full.to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv, "fevs_results.csv", "text/csv")

# Apply filters
if selected_index != 'All':
    df_filtered = df_full[df_full['Index'] == selected_index]
    pivot_filtered = df_filtered.pivot(index='Question', columns='Year', values='Positive')
else:
    df_filtered = df_full
    pivot_filtered = pivot

# Main content
st.title("📊 FEVS Analytics Dashboard")
st.markdown("### Federal Employee Viewpoint Survey | 3-Year Analysis (2023-2025)")

# Executive Scorecard LUMP THIS WITH THE BANNER
st.markdown("---")
st.subheader("📈 Executive Scorecard")

col1, col2, col3, col4, col5 = st.columns(5)

# Calculate key metrics # LUMP THIS WITH THE SCORECARD
total_responses = len(df_main)
response_growth = (df_main[df_main['FY']==2025].shape[0] - df_main[df_main['FY']==2023].shape[0]) / df_main[df_main['FY']==2023].shape[0] * 100
overall_positive = pivot_filtered.mean().mean()
positive_trend = pivot_filtered[2025].mean() - pivot_filtered[2023].mean()
num_strengths = len(pivot_filtered[pivot_filtered.min(axis=1) >= strength_threshold])

col1.metric("Total Responses", f"{total_responses:,}", f"+{response_growth:.1f}%")
col2.metric("Overall Positive", f"{overall_positive:.1f}%", f"{positive_trend:+.1f}%")
col3.metric("Survey Questions", len(pivot_filtered))
col4.metric(f"Strengths (≥{strength_threshold}%)", num_strengths)
col5.metric("Participation Rate", f"{(total_responses/10000*100):.1f}%")  # Assuming 10k target

# Index Performance Gauges
st.markdown("---")
st.subheader("🎯 Index Performance Overview")

index_performance = df_filtered.groupby('Index')['Positive'].mean().sort_values(ascending=False)

fig_gauges = make_subplots(
    rows=1, cols=min(4, len(index_performance)),
    specs=[[{'type': 'indicator'}] * min(4, len(index_performance))],
    subplot_titles=list(index_performance.index[:4])
)

for i, (idx, val) in enumerate(index_performance.head(4).items()):
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=val,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': 'green' if val >= 70 else 'orange' if val >= 60 else 'red'},
               'steps': [
                   {'range': [0, 40], 'color': "lightgray"},
                   {'range': [40, 70], 'color': "lightblue"},
                   {'range': [70, 100], 'color': "lightgreen"}]
               }
    ), row=1, col=i+1)

fig_gauges.update_layout(height=250, showlegend=False)
st.plotly_chart(fig_gauges, use_container_width=True)

# Key Insights Section
st.markdown("---")
st.subheader("💡 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:
    pivot_filtered['Trend'] = pivot_filtered[2025] - pivot_filtered[2023]
    top_improved = pivot_filtered.nlargest(1, 'Trend').reset_index().merge(q_map, on='Question').iloc[0]
    st.info(f"**🚀 Most Improved**\n\n{top_improved['Question']}: +{top_improved['Trend']:.1f}%\n\n*{top_improved['Item.Text'][:80]}...*")

with col2:
    top_performer = pivot_filtered.mean(axis=1).nlargest(1).reset_index()
    top_q = top_performer.merge(q_map, on='Question').iloc[0] # wrong stats
    st.success(f"**⭐ Top Performer**\n\n{top_q['Question']}: {pivot_filtered.loc[top_q['Question']].mean()}%\n\n*{top_q['Item.Text'][:80]}...*")

with col3:
    if len(pivot_filtered[pivot_filtered.min(axis=1) < 60]) > 0: # wrong stats
        needs_attention = pivot_filtered[pivot_filtered.min(axis=1) < 60].mean(axis=1).nsmallest(1).reset_index()
        attn_q = needs_attention.merge(q_map, on='Question').iloc[0]
        st.warning(f"**⚠️ Needs Attention**\n\n{attn_q['Question']}: {pivot_filtered.loc[attn_q['Question']].mean():.1f}%\n\n*{attn_q['Item.Text'][:80]}...*")
    else:
        st.success("**✅ No Critical Issues**\n\nAll questions above 60% threshold")

# Performance Charts
st.markdown("---")
col1, col2 = st.columns(2)

# Add question titles
with col1:
    st.subheader("🏆 Top 10 Performers") # reverse order
    pivot_filtered['Avg'] = pivot_filtered[[2023, 2024, 2025]].mean(axis=1)
    top10 = pivot_filtered.nlargest(10, 'Avg').reset_index().merge(q_map, on='Question')
    fig1 = px.bar(top10, x='Avg', y='Question', orientation='h',
                  color='Avg', color_continuous_scale='Greens',
                  labels={'Avg': '% Positive', 'Question': ''})
    fig1.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📉 Areas for Improvement") # reverse order
    bottom10 = pivot_filtered.nsmallest(10, 'Avg').reset_index().merge(q_map, on='Question')
    fig2 = px.bar(bottom10, x='Avg', y='Question', orientation='h',
                  color='Avg', color_continuous_scale='Reds_r',
                  labels={'Avg': '% Positive', 'Question': ''})
    fig2.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig2, use_container_width=True)

# Index Breakdown / USE BAR CHARTS. PIE CHARTS ARE HARD TO READ
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Index & Sub-Index Performance")
    index_breakdown = df_filtered.groupby(['Index', 'Sub.Index'])['Positive'].mean().reset_index()
    fig3 = px.sunburst(index_breakdown, path=['Index', 'Sub.Index'], values='Positive',
                       color='Positive', color_continuous_scale='RdYlGn',
                       color_continuous_midpoint=65)
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)

# make this into int or categorical
with col2:
    st.subheader("📈 3-Year Trend")
    yearly_avg = df_filtered.groupby('Year')['Positive'].mean()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=yearly_avg.index, y=yearly_avg.values,
                              mode='lines+markers', line=dict(width=3, color='blue'),
                              marker=dict(size=12)))
    fig4.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Target: 70%")
    fig4.update_layout(yaxis_title="% Positive", height=250, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("📊 Response Mix")
    avg_by_year = df_filtered.groupby('Year')[['Positive', 'Neutral', 'Negative']].mean()
    fig5 = px.bar(avg_by_year, barmode='stack',
                  color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
    fig5.update_layout(height=250, showlegend=True, yaxis_title="% Distribution")
    st.plotly_chart(fig5, use_container_width=True)

# Heatmap VERY HARD TO SEE THE DIFFERENCE
# INCLUDE THE QUESTION TEXT.Q10 and Q90 is very hard to know. Maybe a hover effect
st.markdown("---")
st.subheader("🔥 Question Performance Heatmap")
pivot_display = pivot_filtered[[2023, 2024, 2025]].sort_values(by=2025, ascending=False)
fig6 = px.imshow(pivot_display.values, x=[2023, 2024, 2025], y=pivot_display.index,
                 color_continuous_scale='RdYlGn', aspect='auto', height=700,
                 labels={'color': '% Positive'})
st.plotly_chart(fig6, use_container_width=True)

# Question Drilldown
if selected_question != 'None':
    st.markdown("---")
    st.subheader(f"🔍 Question Detail: {selected_question}")

    q_info = q_map[q_map['Question'] == selected_question].iloc[0]
    q_data = df_results[df_results['Question'] == selected_question]

    st.markdown(f"**{q_info['Item.Text']}**")
    st.caption(f"📁 Index: {q_info['Index']} › {q_info['Sub.Index']}")

    col1, col2, col3, col4 = st.columns(4)
    avg_pos = q_data['Positive'].mean()
    trend = pivot.loc[selected_question, 2025] - pivot.loc[selected_question, 2023]
    col1.metric("Avg Positive", f"{avg_pos:.1f}%")
    col2.metric("2023", f"{pivot.loc[selected_question, 2023]:.1f}%")
    col3.metric("2024", f"{pivot.loc[selected_question, 2024]:.1f}%")
    col4.metric("2025", f"{pivot.loc[selected_question, 2025]:.1f}%", f"{trend:+.1f}%")

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Positive'], mode='lines+markers',
                             name='Positive', line=dict(color='green', width=4), marker=dict(size=12)))
    fig7.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Neutral'], mode='lines+markers',
                             name='Neutral', line=dict(color='gray', width=3), marker=dict(size=10)))
    fig7.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Negative'], mode='lines+markers',
                             name='Negative', line=dict(color='red', width=3), marker=dict(size=10)))
    fig7.update_layout(hovermode='x unified', height=350, yaxis_title="Percentage")
    st.plotly_chart(fig7, use_container_width=True)

# Action Recommendations
st.markdown("---")
st.subheader("🎯 Recommended Actions")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Priority Focus Areas:**")
    focus_areas = pivot_filtered[pivot_filtered.max(axis=1) < 60].mean(axis=1).nsmallest(3).reset_index()

    if len(focus_areas) > 0:
        focus_areas = focus_areas.merge(q_map, on='Question')
        for idx, row in focus_areas.iterrows():
            st.write(f"• **{row['Index']}** - {row['Sub.Index']}")
    else:
        # Show bottom performers even if above 60%
        focus_areas = pivot_filtered.mean(axis=1).nsmallest(3).reset_index()
        focus_areas = focus_areas.merge(q_map, on='Question')
        st.caption("*(Lowest performers - all above improvement threshold)*")
        for idx, row in focus_areas.iterrows():
            st.write(f"• **{row['Index']}** - {row['Sub.Index']}")

with col2:
    st.markdown("**Maintain Strengths In:**")
    maintain = pivot_filtered[pivot_filtered.min(axis=1) >= 70].mean(axis=1).nlargest(3).reset_index()

    if len(maintain) > 0:
        maintain = maintain.merge(q_map, on='Question')
        for idx, row in maintain.iterrows():
            st.write(f"• **{row['Index']}** - {row['Sub.Index']}")
    else:
        # Show top performers even if below 70%
        maintain = pivot_filtered.mean(axis=1).nlargest(3).reset_index()
        maintain = maintain.merge(q_map, on='Question')
        st.caption("*(Top performers - below strength threshold)*")
        for idx, row in maintain.iterrows():
            st.write(f"• **{row['Index']}** - {row['Sub.Index']}")