import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
                results.append({'Question': q, 'Year': str(year), 'Positive': pos, 'Neutral': neu, 'Negative': neg})
    return pd.DataFrame(results)


df_results = calc_data(df_main)
# Convert Year to categorical with proper ordering
df_results['Year'] = pd.Categorical(df_results['Year'], categories=['2023', '2024', '2025'], ordered=True)

pivot = df_results.pivot(index='Question', columns='Year', values='Positive')
q_map = df_questions[['Item.ID', 'Item.Text', 'Index', 'Sub.Index']].rename(columns={'Item.ID': 'Question'})
df_full = df_results.merge(q_map, on='Question')


# Helper function to truncate text
def truncate_text(text, max_length=100):
    return text[:max_length] + "..." if len(text) > max_length else text


# Sidebar
st.sidebar.title("🔧 Controls")
selected_index = st.sidebar.selectbox("Filter by Index:", ['All'] + sorted(df_full['Index'].dropna().unique().tolist()))
strength_threshold = st.sidebar.slider("Strength (%) (lower-bound):", 60, 90, 70, 5)
weakness_threshold = st.sidebar.slider("Improvement (%) (higher-bound):", 30, 60, 60, 5)

st.sidebar.markdown("---")
question_cols = sorted(pivot.index.tolist(), key=lambda x: int(x[1:]))
selected_question = st.sidebar.selectbox("Question Drilldown:", ['None'] + question_cols)

if st.sidebar.button("📥 Download Results"):
    csv = df_full.to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv, "fevs_results.csv", "text/csv")

# Apply filters
if selected_index != 'All':
    df_filtered = df_full[df_full['Index'] == selected_index]
    pivot_filtered = df_filtered.pivot(index='Question', columns='Year', values='Positive')
else:
    df_filtered = df_full
    pivot_filtered = pivot

# Title
st.title("📊 FEVS Analytics | 2023-2025")

col1, col2, col3, col4, col5 = st.columns(5)

total_responses = len(df_main)
response_growth = (df_main[df_main['FY'] == 2025].shape[0] - df_main[df_main['FY'] == 2023].shape[0]) / \
                  df_main[df_main['FY'] == 2023].shape[0] * 100
overall_positive = pivot_filtered.mean().mean()
positive_trend = pivot_filtered['2025'].mean() - pivot_filtered['2023'].mean()
num_strengths = len(pivot_filtered[pivot_filtered.min(axis=1) >= strength_threshold])

col1.metric("Responses", f"{total_responses:,}", f"+{response_growth:.1f}%")
col2.metric("Avg Positive", f"{overall_positive:.1f}%", f"{positive_trend:+.1f}%")
col3.metric("Questions", len(pivot_filtered))
col4.metric(f"Strengths ≥{strength_threshold}%", num_strengths)
col5.metric("Participation", f"{(total_responses / 10000 * 100):.1f}%")

# Compact Trend Charts
col_trend1, col_trend2 = st.columns(2)

with col_trend1:
    st.markdown("**📈 3-Year Trend**")
    yearly_avg = df_filtered.groupby('Year')['Positive'].mean()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=yearly_avg.index, y=yearly_avg.values,
                              mode='lines+markers', line=dict(width=3, color='blue'),
                              marker=dict(size=10)))
    fig4.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Target: 70%",
                   annotation_position="right")
    fig4.update_layout(yaxis_title="% Positive", height=200, showlegend=False,
                       margin=dict(l=0, r=0, t=0, b=0))
    fig4.update_xaxes(type='category')
    st.plotly_chart(fig4, use_container_width=True)

with col_trend2:
    st.markdown("**📊 Response Mix**")
    avg_by_year = df_filtered.groupby('Year')[['Positive', 'Neutral', 'Negative']].mean()
    fig5 = px.bar(avg_by_year, barmode='stack',
                  color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
    fig5.update_layout(height=200, showlegend=True, yaxis_title="% Distribution",
                       margin=dict(l=0, r=0, t=0, b=0),
                       legend=dict(orientation="h", yanchor="bottom", y=1.2, xanchor="center", x=0.5))
    fig5.update_xaxes(type='category')
    st.plotly_chart(fig5, use_container_width=True)

# Executive Summary - Key Insights
st.markdown("##### Key Insights")
col1, col2 = st.columns(2)


with col1:
    pivot_filtered['Avg_Score'] = pivot_filtered[['2023', '2024', '2025']].mean(axis=1)
    top_performer = pivot_filtered.nlargest(1, 'Avg_Score')
    top_q_id = top_performer.index[0]
    top_q_val = top_performer['Avg_Score'].values[0]
    top_q = q_map[q_map['Question'] == top_q_id].iloc[0]
    st.success(f"**⭐ Top Performer**  \n{top_q_id}: **{top_q_val:.1f}%**  \n*{truncate_text(top_q['Item.Text'])}*")

with col2:
    bottom_performer = pivot_filtered.nsmallest(1, 'Avg_Score')
    btm_q_id = bottom_performer.index[0]
    btm_q_val = bottom_performer['Avg_Score'].values[0]
    btm_q = q_map[q_map['Question'] == btm_q_id].iloc[0]
    if btm_q_val < 60:
        st.warning(
            f"**⚠️ Needs Attention**  \n{btm_q_id}: **{btm_q_val:.1f}%**  \n*{truncate_text(btm_q['Item.Text'])}*")
    else:
        st.success(f"**✅ Lowest (Still Good)**  \n{btm_q_id}: **{btm_q_val:.1f}%**  \n*All questions above 60%*")


# Top/Bottom Performers
st.markdown("##### Performance Analysis")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**🏆 Top 10 Performers**")
    pivot_filtered['Avg'] = pivot_filtered[['2023', '2024', '2025']].mean(axis=1)
    top10 = pivot_filtered.nlargest(10, 'Avg').reset_index().merge(q_map, on='Question')
    top10['Label'] = top10['Question'] + ': ' + top10['Item.Text'].apply(lambda x: truncate_text(x, 100))
    top10 = top10.sort_values('Avg')  # Ascending for horizontal bar

    fig1 = px.bar(top10, x='Avg', y='Label', orientation='h',
                  color='Avg', color_continuous_scale='Greens',
                  labels={'Avg': '% Positive', 'Label': ''})
    fig1.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=10, b=0))
    fig1.update_xaxes(showticklabels=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("**📉 Areas for Improvement**")
    bottom10 = pivot_filtered.nsmallest(10, 'Avg').reset_index().merge(q_map, on='Question')
    bottom10['Label'] = bottom10['Question'] + ': ' + bottom10['Item.Text'].apply(lambda x: truncate_text(x, 100))
    bottom10 = bottom10.sort_values('Avg', ascending=False)  # Descending for horizontal bar

    fig2 = px.bar(bottom10, x='Avg', y='Label', orientation='h',
                  color='Avg', color_continuous_scale='Reds_r',
                  labels={'Avg': '% Positive', 'Label': ''})
    fig2.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=10, b=0))
    fig2.update_xaxes(showticklabels=False)
    st.plotly_chart(fig2, use_container_width=True)

# Gauge Charts for Index Performance
st.markdown("##### Index Performance - 3 Years Average")
index_performance = df_filtered.groupby('Index')['Positive'].mean().sort_values(ascending=False)

num_gauges = min(5, len(index_performance))
fig_gauges = make_subplots(
    rows=1, cols=num_gauges,
    specs=[[{'type': 'indicator'}] * num_gauges],
    subplot_titles=[truncate_text(idx, 20) for idx in index_performance.index[:num_gauges]]
)

for i, (idx, val) in enumerate(index_performance.head(num_gauges).items()):
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=val,
        number={'suffix': '%', 'font': {'size': 16}},
        gauge={'axis': {'range': [0, 100], 'tickfont': {'size': 10}, 'tickmode': 'array', 'tickvals': [20, 40, 60, 80, 100]},
               'bar': {'color': 'green' if val >= 70 else 'orange' if val >= 60 else 'red', 'thickness': 0.7},
               'steps': [
                   {'range': [0, 40], 'color': "lightgray"},
                   {'range': [40, 70], 'color': "lightblue"},
                   {'range': [70, 100], 'color': "lightgreen"}]
               }
    ), row=1, col=i + 1)

fig_gauges.update_layout(height=180, showlegend=False, margin=dict(t=40, b=0, l=20, r=20))
st.plotly_chart(fig_gauges, use_container_width=True)



# Sub-Index Breakdown
st.markdown("##### Index & Sub-Index Breakdown - 3 Years Average")
index_breakdown = df_filtered.groupby(['Index', 'Sub.Index'])['Positive'].mean().reset_index()
# Calculate average per Index for sorting
index_avg = df_filtered.groupby('Index')['Positive'].mean().sort_values(ascending=False)
# Create ordered category for Index based on performance
index_breakdown['Index'] = pd.Categorical(index_breakdown['Index'], categories=index_avg.index)
index_breakdown = index_breakdown.sort_values(['Index', 'Positive'], ascending=[False, False])

fig3 = px.bar(index_breakdown, x='Positive', y='Sub.Index', color='Index',
              orientation='h',
              labels={'Positive': '% Positive', 'Sub.Index': 'Sub-Index'},
              color_discrete_sequence=px.colors.qualitative.Set2)
fig3.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0),
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig3, use_container_width=True)



# Improved Heatmap with Question Text
st.markdown("##### Question Performance Heatmap")
pivot_display = pivot_filtered[['2023', '2024', '2025']].sort_values(by='2025', ascending=False)
pivot_with_text = pivot_display.reset_index().merge(q_map, on='Question')
pivot_with_text['Display'] = pivot_with_text['Question'] + ': ' + pivot_with_text['Item.Text'].apply(
    lambda x: truncate_text(x, 60))

# Create heatmap with hover text
fig6 = go.Figure(data=go.Heatmap(
    z=pivot_display.values,
    x=['2023', '2024', '2025'],  # Use strings for categorical
    y=pivot_with_text['Display'],
    colorscale='RdYlGn',
    text=pivot_display.values.round(1),
    texttemplate='%{text}%',
    textfont={"size": 8},
    hovertemplate='Year: %{x}<br>%{y}<br>Positive: %{z:.1f}%<extra></extra>',
    colorbar=dict(title='% Positive')
))
fig6.update_layout(height=max(400, len(pivot_display) * 20),
                   margin=dict(l=0, r=0, t=10, b=0),
                   xaxis_title="", yaxis_title="")
fig6.update_xaxes(type='category')  # Force categorical axis
st.plotly_chart(fig6, use_container_width=True)

# Question Drilldown
if selected_question != 'None':
    st.markdown("---")
    st.markdown(f"##### 🔍 Question Detail: {selected_question}")

    q_info = q_map[q_map['Question'] == selected_question].iloc[0]
    q_data = df_results[df_results['Question'] == selected_question].copy()
    q_data['Year'] = q_data['Year'].astype(str)  # Ensure string type

    st.markdown(f"**{q_info['Item.Text']}**")
    st.caption(f"📁 {q_info['Index']} › {q_info['Sub.Index']}")

    col1, col2, col3, col4 = st.columns(4)
    avg_pos = q_data['Positive'].mean()
    trend = pivot.loc[selected_question, '2025'] - pivot.loc[selected_question, '2023']
    col1.metric("Avg", f"{avg_pos:.1f}%")
    col2.metric("2023", f"{pivot.loc[selected_question, '2023']:.1f}%")
    col3.metric("2024", f"{pivot.loc[selected_question, '2024']:.1f}%")
    col4.metric("2025", f"{pivot.loc[selected_question, '2025']:.1f}%", f"{trend:+.1f}%")

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Positive'], mode='lines+markers',
                              name='Positive', line=dict(color='green', width=3), marker=dict(size=10)))
    fig7.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Neutral'], mode='lines+markers',
                              name='Neutral', line=dict(color='gray', width=2), marker=dict(size=8)))
    fig7.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Negative'], mode='lines+markers',
                              name='Negative', line=dict(color='red', width=2), marker=dict(size=8)))
    fig7.update_layout(hovermode='x unified', height=280, yaxis_title="Percentage",
                       margin=dict(l=0, r=0, t=10, b=0))
    fig7.update_xaxes(type='category')  # Force categorical axis
    st.plotly_chart(fig7, use_container_width=True)
