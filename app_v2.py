import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("FEVS Analytics Dashboard")

# Load data
df_main = pd.read_excel('data/fevs_sample_data_3FYs_DataSet_1.xlsx', sheet_name='fevs_sample_data_3FYs_Set1')
df_questions = pd.read_excel('data/fevs_sample_data_3FYs_DataSet_1.xlsx', sheet_name='Index-Qns-Map', skiprows=8)

# Calculate percentages
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

df_results = pd.DataFrame(results)
pivot = df_results.pivot(index='Question', columns='Year', values='Positive')

# Identify patterns
strengths = pivot[pivot.min(axis=1) >= 70].reset_index()
improvement = pivot[pivot.min(axis=1) < 60].reset_index()

# Add question text
q_map = df_questions[['Item.ID', 'Item.Text', 'Index']].rename(columns={'Item.ID': 'Question'})
strengths = strengths.merge(q_map, on='Question')
improvement = improvement.merge(q_map, on='Question')
df_full = df_results.merge(q_map, on='Question')

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Responses", f"{len(df_main):,}")
col2.metric("Questions", len(question_cols))
col3.metric("Strengths (≥70%)", len(strengths))
col4.metric("Need Attention (<60%)", len(improvement))

# Chart 1: Top 10 Strengths
st.subheader("Top 10 Strengths")
strengths['Avg'] = strengths[[2023, 2024, 2025]].mean(axis=1)
top10 = strengths.nlargest(10, 'Avg')
fig1 = px.bar(top10, x='Avg', y='Item.Text', orientation='h', color='Avg',
              color_continuous_scale='Greens')
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Bottom 10 Questions
st.subheader("Bottom 10 Questions")
improvement['Avg'] = improvement[[2023, 2024, 2025]].mean(axis=1)
bottom10 = improvement.nsmallest(10, 'Avg')
fig2 = px.bar(bottom10, x='Avg', y='Item.Text', orientation='h', color='Avg',
              color_continuous_scale='Reds_r')
st.plotly_chart(fig2, use_container_width=True)

# Chart 3: Trend Heatmap
st.subheader("3-Year Trend Heatmap (All Questions)")
pivot_display = pivot.sort_values(by=2025, ascending=False)
fig3 = px.imshow(pivot_display.values, x=[2023, 2024, 2025], y=pivot_display.index,
                 color_continuous_scale='RdYlGn', aspect='auto',
                 labels={'x': 'Year', 'y': 'Question', 'color': '% Positive'})
st.plotly_chart(fig3, use_container_width=True)

# Chart 4: Performance by Index
st.subheader("Performance by Category")
by_index = df_full.groupby('Index')['Positive'].mean().sort_values(ascending=False)
fig4 = px.bar(by_index, orientation='h', color=by_index.values,
              color_continuous_scale='Blues')
st.plotly_chart(fig4, use_container_width=True)

# Chart 5: Distribution of Responses
st.subheader("Overall Response Distribution")
avg_by_year = df_results.groupby('Year')[['Positive', 'Neutral', 'Negative']].mean()
fig5 = px.bar(avg_by_year, barmode='stack',
              color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
st.plotly_chart(fig5, use_container_width=True)

# Chart 6: Trend Lines for Selected Questions
st.subheader("Question Trend Analysis")
selected = st.multiselect("Select questions to compare:", question_cols, default=['Q1', 'Q2', 'Q3'])
if selected:
    trend_data = df_results[df_results['Question'].isin(selected)]
    fig6 = px.line(trend_data, x='Year', y='Positive', color='Question', markers=True)
    st.plotly_chart(fig6, use_container_width=True)

# Chart 7: Biggest Movers
st.subheader("Biggest Changes (2023 → 2025)")
pivot['Change'] = pivot[2025] - pivot[2023]
col1, col2 = st.columns(2)

with col1:
    st.write("**Most Improved**")
    improved = pivot.nlargest(5, 'Change').reset_index()
    improved = improved.merge(q_map, on='Question')
    fig7 = px.bar(improved, x='Change', y='Question', orientation='h', color='Change',
                  color_continuous_scale='Greens')
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    st.write("**Most Declined**")
    declined = pivot.nsmallest(5, 'Change').reset_index()
    declined = declined.merge(q_map, on='Question')
    fig8 = px.bar(declined, x='Change', y='Question', orientation='h', color='Change',
                  color_continuous_scale='Reds_r')
    st.plotly_chart(fig8, use_container_width=True)

# Chart 8: Year-over-Year Comparison
st.subheader("Year-over-Year Comparison")
yearly_avg = df_results.groupby('Year')['Positive'].mean()
fig9 = go.Figure()
fig9.add_trace(go.Bar(x=yearly_avg.index, y=yearly_avg.values, marker_color='lightblue'))
fig9.update_layout(yaxis_title="Average % Positive")
st.plotly_chart(fig9, use_container_width=True)

# Chart 9: Response Volume by Year
st.subheader("Response Volume Trend")
volume_by_year = df_main['FY'].value_counts().sort_index()
fig10 = px.line(x=volume_by_year.index, y=volume_by_year.values, markers=True)
fig10.update_layout(xaxis_title="Year", yaxis_title="Number of Responses")
st.plotly_chart(fig10, use_container_width=True)