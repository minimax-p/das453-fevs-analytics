import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="FEVS Analytics")


# Load data
@st.cache_data
def load_data():
    df_main = pd.read_excel('data/fevs_sample_data_3FYs_DataSet_1.xlsx', sheet_name='fevs_sample_data_3FYs_Set1')
    df_questions = pd.read_excel('data/fevs_sample_data_3FYs_DataSet_1.xlsx', sheet_name='Index-Qns-Map', skiprows=8)
    return df_main, df_questions


df_main, df_questions = load_data()


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

# Sidebar filters
st.sidebar.title("🔧 Drill-Down Settings")

# Filter by Index
indices = ['All'] + sorted(df_full['Index'].dropna().unique().tolist())
selected_index = st.sidebar.selectbox("Filter by Category:", indices)

# Filter by threshold
strength_threshold = st.sidebar.slider("Strength Threshold (%):", 60, 90, 70)
weakness_threshold = st.sidebar.slider("Improvement Threshold (%):", 30, 60, 60)

# Question selector
question_cols = sorted(pivot.index.tolist(), key=lambda x: int(x[1:]))
selected_question = st.sidebar.selectbox("Select Question for Detail:", ['None'] + question_cols)

# Year range
year_range = st.sidebar.multiselect("Years to Display:", [2023, 2024, 2025], default=[2023, 2024, 2025])

st.sidebar.markdown("---")
st.sidebar.markdown("**Export Options**")
if st.sidebar.button("Download Results CSV"):
    csv = df_full.to_csv(index=False)
    st.sidebar.download_button("Download", csv, "fevs_results.csv", "text/csv")

# Apply filters
if selected_index != 'All':
    df_filtered = df_full[df_full['Index'] == selected_index]
    pivot_filtered = df_filtered.pivot(index='Question', columns='Year', values='Positive')
else:
    df_filtered = df_full
    pivot_filtered = pivot

if year_range:
    df_filtered = df_filtered[df_filtered['Year'].isin(year_range)]

# Main dashboard
st.title("FEVS Analytics Dashboard")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Responses", f"{len(df_main):,}")
col2.metric("Questions", len(pivot_filtered))
col3.metric(f"Strengths (≥{strength_threshold}%)",
            len(pivot_filtered[pivot_filtered.min(axis=1) >= strength_threshold]))
col4.metric(f"Need Attention (<{weakness_threshold}%)",
            len(pivot_filtered[pivot_filtered.min(axis=1) < weakness_threshold]))

# Layout: 2 columns
col1, col2 = st.columns(2)

with col1:
    # Top performers
    st.subheader("Top 10 Performers")
    pivot_filtered['Avg'] = pivot_filtered[year_range].mean(axis=1)
    top10 = pivot_filtered.nlargest(10, 'Avg').reset_index().merge(q_map, on='Question')
    fig1 = px.bar(top10, x='Avg', y='Question', orientation='h', color='Avg', color_continuous_scale='Greens')
    fig1.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Bottom performers
    st.subheader("Bottom 10 Performers")
    bottom10 = pivot_filtered.nsmallest(10, 'Avg').reset_index().merge(q_map, on='Question')
    fig2 = px.bar(bottom10, x='Avg', y='Question', orientation='h', color='Avg', color_continuous_scale='Reds_r')
    fig2.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig2, use_container_width=True)

# Full width charts
col1, col2 = st.columns(2)

with col1:
    # Performance by category
    st.subheader("Performance by Category")
    by_index = df_filtered.groupby('Index')['Positive'].mean().sort_values(ascending=False)
    fig3 = px.bar(by_index, orientation='h', color=by_index.values, color_continuous_scale='Blues')
    fig3.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    # Response distribution
    st.subheader("Response Distribution")
    avg_by_year = df_filtered.groupby('Year')[['Positive', 'Neutral', 'Negative']].mean()
    fig4 = px.bar(avg_by_year, barmode='stack',
                  color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
    fig4.update_layout(height=300)
    st.plotly_chart(fig4, use_container_width=True)

# Heatmap
st.subheader("Trend Heatmap")
pivot_display = pivot_filtered[year_range].sort_values(by=year_range[0] if year_range else 2023, ascending=False)
fig5 = px.imshow(pivot_display.values, x=year_range, y=pivot_display.index,
                 color_continuous_scale='RdYlGn', aspect='auto', height=600)
st.plotly_chart(fig5, use_container_width=True)

# Question detail (if selected)
if selected_question != 'None':
    st.markdown("---")
    st.subheader(f"🔍 Question Detail: {selected_question}")

    q_info = q_map[q_map['Question'] == selected_question].iloc[0]
    q_data = df_results[df_results['Question'] == selected_question]

    st.write(f"**{q_info['Item.Text']}**")
    st.caption(f"Index: {q_info['Index']} > {q_info['Sub.Index']}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Positive", f"{q_data['Positive'].mean():.1f}%")
    col2.metric("2023", f"{pivot.loc[selected_question, 2023]:.1f}%")
    col3.metric("2024", f"{pivot.loc[selected_question, 2024]:.1f}%")
    col4.metric("2025", f"{pivot.loc[selected_question, 2025]:.1f}%")

    # Trend chart
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Positive'], mode='lines+markers',
                              name='Positive', line=dict(color='green', width=3)))
    fig6.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Neutral'], mode='lines+markers',
                              name='Neutral', line=dict(color='gray', width=2)))
    fig6.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Negative'], mode='lines+markers',
                              name='Negative', line=dict(color='red', width=2)))
    fig6.update_layout(hovermode='x unified', height=300)
    st.plotly_chart(fig6, use_container_width=True)