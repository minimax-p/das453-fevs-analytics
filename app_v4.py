# /mnt/data/app_v4_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="FEVS Analytics", page_icon="📊")

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    df_main = pd.read_excel('data/fevs_sample_data_3FYs_DataSet_1.xlsx', sheet_name='fevs_sample_data_3FYs_Set1')
    df_questions = pd.read_excel('data/fevs_sample_data_3FYs_DataSet_1.xlsx', sheet_name='Index-Qns-Map', skiprows=8)
    df_indices = pd.read_excel('data/fevs_sample_data_3FYs_DataSet_1.xlsx', sheet_name='Index-Def-Map')
    return df_main, df_questions, df_indices

df_main, df_questions, df_indices = load_data()

# ---------------------------
# Precompute question-level yearly percentages
# ---------------------------
@st.cache_data
def calc_data(df_main):
    results = []
    question_cols = [col for col in df_main.columns if col.startswith('Q')]
    for q in question_cols:
        for year in [2023, 2024, 2025]:
            responses = df_main[df_main['FY'] == year][q]
            total = len(responses)
            if total > 0:
                pos = (responses >= 4).sum() / total * 100
                neu = (responses == 3).sum() / total * 100
                neg = (responses <= 2).sum() / total * 100
                results.append({'Question': q, 'Year': str(year), 'Positive': pos, 'Neutral': neu, 'Negative': neg})
    return pd.DataFrame(results)

df_results = calc_data(df_main)
df_results['Year'] = pd.Categorical(df_results['Year'], categories=['2023', '2024', '2025'], ordered=True)

pivot = df_results.pivot(index='Question', columns='Year', values='Positive')

q_map = df_questions[['Item.ID', 'Item.Text', 'Index', 'Sub.Index']].rename(columns={'Item.ID': 'Question'})
df_full = df_results.merge(q_map, on='Question')

# helper
def truncate_text(text, max_length=100):
    return text[:max_length] + "..." if isinstance(text, str) and len(text) > max_length else text

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.title("Controls")
index_options = ['All'] + sorted(df_full['Index'].unique().tolist())
selected_index = st.sidebar.selectbox("Filter by Index:", index_options)

# Threshold sliders: make them meaningful for coloring & flags
strength_threshold = st.sidebar.slider("Strength threshold (%) (>=):", min_value=50, max_value=90, value=73, step=1)
weakness_threshold = st.sidebar.slider("Weakness threshold (%) (<):", min_value=20, max_value=70, value=65, step=1)

# INDEX FILTERING
# Apply index filter to df_full; then create dynamic question dropdown based on that filter
if selected_index != 'All':
    df_filtered = df_full[df_full['Index'] == selected_index].copy()
    pivot_filtered = df_filtered.pivot(index='Question', columns='Year', values='Positive')
else:
    df_filtered = df_full.copy()
    pivot_filtered = pivot.copy()

# DYNAMIC QUESTION DROPDOWN
# Create dynamic question list for drilldown: only questions present in the filtered index
question_options = ['None']
q_list_in_index = sorted(pivot_filtered.index.tolist(), key=lambda x: int(x[1:]) if x.startswith('Q') and x[1:].isdigit() else x)
question_options += q_list_in_index
selected_question = st.sidebar.selectbox("Question Drilldown:", question_options)

st.sidebar.markdown("---")
# how to make the text smaller?
st.sidebar.markdown("Note: Strength and Weakness thresholds set in sidebar affect color coding and flags throughout the dashboard. The rule is simple: If a question has an average % Positive score greater than or equal to the Strength threshold, it is classified as a Strength. If it has an average % Positive score less than the Weakness threshold, it is classified as a Weakness.")

# ---------------------------
# Title
# ---------------------------
st.title("FEVS Analytics | 2023–2025")  # removed emojis for a cleaner look

# ---------------------------
# Metrics (per selected index)
# --------------------------
# Compute total responses in the context of the selected index:
# count respondents (rows in df_main) that have at least one non-null response among questions in this index.
if selected_index != 'All':
    idx_questions = q_map[q_map['Index'] == selected_index]['Question'].tolist()
    total_responses_idx = df_main[idx_questions].shape[0]
else:
    total_responses_idx = len(df_main)

# Avoid modifying pivot_filtered in place: create a working copy for metric calcs
pf = pivot_filtered.copy()
overall_positive = pf.mean().mean()
num_strengths = (pf.mean(axis=1) >= strength_threshold).sum()
num_weaknesses = (pf.mean(axis=1) < weakness_threshold).sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total responses (people)", f"{total_responses_idx:,}")
c2.metric("Average Positive", f"{overall_positive:.2f}%" if not np.isnan(overall_positive) else "N/A")
c3.metric("Questions in view", len(pf))
c4.metric(f"Strengths (avg ≥ {strength_threshold}%)", int(num_strengths))
c5.metric(f"Weaknesses (avg < {weakness_threshold}%)", int(num_weaknesses))

# Small trend + response mix
st.markdown("### Trends")
col_trend1, col_trend2 = st.columns(2)
with col_trend1:
    yearly_avg = df_filtered.groupby('Year')['Positive'].mean()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=yearly_avg.index, y=yearly_avg.values,
                              mode='lines+markers', line=dict(width=3),
                              marker=dict(size=8)))
    fig4.add_hline(y=strength_threshold, line_dash="dash", line_color="green",
                   annotation_text=f"Strength threshold: {strength_threshold}%")
    fig4.add_hline(y=weakness_threshold, line_dash="dash", line_color="red",
                   annotation_text=f"Weakness threshold: {weakness_threshold}%")
    fig4.update_layout(yaxis_title="% Positive", height=240, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    fig4.update_xaxes(type='category')
    st.plotly_chart(fig4, use_container_width=True)

with col_trend2:
    avg_by_year = df_filtered.groupby('Year')[['Positive', 'Neutral', 'Negative']].mean()
    fig5 = px.bar(avg_by_year, barmode='stack',
                  color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
    fig5.update_layout(height=240, showlegend=True, yaxis_title="% Distribution",
                       margin=dict(l=0, r=0, t=0, b=0),
                       legend=dict(orientation="h", yanchor="bottom", y=1.2, xanchor="center", x=0.5),
                       legend_title_text=None)
    fig5.update_xaxes(type='category')
    st.plotly_chart(fig5, use_container_width=True)

# ---------------------------
# Key Insights & Performance Analysis
# ---------------------------
st.markdown("### Key insights")
pf = pivot_filtered.copy()
# compute trend and averages safely
if '2023' in pf.columns and '2025' in pf.columns:
    pf['Trend'] = pf['2025'] - pf['2023']
else:
    pf['Trend'] = np.nan

pf['Avg_Score'] = pf[[c for c in ['2023', '2024', '2025'] if c in pf.columns]].mean(axis=1)

c1, c2, c3 = st.columns(3)
# Top performer
top_performer_row = pf['Avg_Score'].idxmax()
top_performer = pf.loc[top_performer_row]
q_info = q_map[q_map['Question'] == top_performer_row].iloc[0]
c1.success(
    f"**⭐ Top Performer (avg)**:\n\n{top_performer_row} ({top_performer['Avg_Score']:.1f}%)\n\n{truncate_text(q_info['Item.Text'], 120)}")

# Bottom performer (use weakness_threshold to flag)
bottom_performer_row = pf['Avg_Score'].idxmin()
bottom_performer = pf.loc[bottom_performer_row]
q_info = q_map[q_map['Question'] == bottom_performer_row].iloc[0]
if bottom_performer['Avg_Score'] < weakness_threshold:
    c2.error(
        f"**🚨 Needs Attention (avg < {weakness_threshold}%)**:\n\n{bottom_performer_row} ({bottom_performer['Avg_Score']:.1f}%)\n\n{truncate_text(q_info['Item.Text'], 60)}")
else:
    c2.info(
        f"**Lowest but above threshold (>{weakness_threshold}%)**:\n\n{bottom_performer_row} ({bottom_performer['Avg_Score']:.1f}%)\n\n{truncate_text(q_info['Item.Text'], 60)}"
    )
# Most improved
top_improved_row = pf['Trend'].idxmax()
top_improved = pf.loc[top_improved_row]
q_info = q_map[q_map['Question'] == top_improved_row].iloc[0]
c3.info(
    f"**🚀 Most Improved (2025 vs 2023)**:\n\n{top_improved_row} ({top_improved['Trend']:+.1f}%)\n\n{truncate_text(q_info['Item.Text'], 60)}")



# Performance Analysis (Top/Bottom)
st.markdown("### Performance analysis")
pf = pivot_filtered.copy()
pf['Avg'] = pf[[c for c in ['2023','2024','2025'] if c in pf.columns]].mean(axis=1)

# classify by thresholds
pf['Category'] = pf['Avg'].apply(lambda v: "Strength" if v >= strength_threshold else ("Weakness" if v <= weakness_threshold else "Neutral"))
color_map = {"Strength":"green", "Neutral":"gray", "Weakness":"red"}



n_q = len(pf)

if n_q > 20:
    col_left, col_right = st.columns(2)

    # Top 10
    top10 = pf.nlargest(10, 'Avg').reset_index().merge(q_map, on='Question')
    top10['Label'] = top10['Question'] + ": " + top10['Item.Text'].apply(lambda x: truncate_text(x,30))
    top10 = top10.sort_values('Avg')
    fig1 = px.bar(top10, x='Avg', y='Label', orientation='h', color='Category', color_discrete_map=color_map, labels={'Avg':'% Positive','Label':''})
    fig1.add_vline(x=strength_threshold, line_dash='dash', line_color='green')
    fig1.add_vline(x=weakness_threshold, line_dash='dash', line_color='red')
    fig1.update_traces(text=top10['Avg'].round(1).astype(str)+'%', textposition='inside')
    fig1.update_layout(showlegend=False, height=350, margin=dict(l=0,r=0,t=10,b=0))
    fig1.update_xaxes(showticklabels=False)
    col_left.markdown("Top 10 Strengths")
    col_left.plotly_chart(fig1, use_container_width=True)

    # Bottom 10
    bottom10 = pf.nsmallest(10, 'Avg').reset_index().merge(q_map, on='Question')
    bottom10['Label'] = bottom10['Question'] + ": " + bottom10['Item.Text'].apply(lambda x: truncate_text(x,30))
    bottom10 = bottom10.sort_values('Avg', ascending=True)
    fig2 = px.bar(bottom10, x='Avg', y='Label', orientation='h', color='Category', color_discrete_map=color_map, labels={'Avg':'% Positive','Label':''})
    fig2.add_vline(x=strength_threshold, line_dash='dash', line_color='green')
    fig2.add_vline(x=weakness_threshold, line_dash='dash', line_color='red')
    fig2.update_traces(text=bottom10['Avg'].round(1).astype(str)+'%', textposition='inside')
    fig2.update_layout(showlegend=False, height=350, margin=dict(l=0,r=0,t=10,b=0))
    fig2.update_xaxes(showticklabels=False)
    col_right.markdown("Bottom 10 Weaknesses")
    col_right.plotly_chart(fig2, use_container_width=True)

else:
    ranked = pf.reset_index().merge(q_map, on='Question').sort_values('Avg')
    ranked['LabelShort'] = ranked['Question'] + ": " + ranked['Item.Text'].apply(lambda x: truncate_text(x,30))
    fig_all = px.bar(ranked, x='Avg', y='LabelShort', orientation='h', color='Category', color_discrete_map=color_map, labels={'Avg':'% Positive','LabelShort':''})
    fig_all.add_vline(x=strength_threshold, line_dash='dash', line_color='green')
    fig_all.add_vline(x=weakness_threshold, line_dash='dash', line_color='red')
    fig_all.update_traces(text=ranked['Avg'].round(1).astype(str)+'%', textposition='inside')
    fig_all.update_layout(showlegend=False, height=480, margin=dict(l=0,r=0,t=10,b=0))
    fig_all.update_xaxes(showticklabels=False)
    st.markdown("**All questions (ranked) — fewer than 20 questions available for this index**")
    st.plotly_chart(fig_all, use_container_width=True)

# ---------------------------
# Index tab: Gauges & Sub-index breakdown
# ---------------------------
st.markdown("### Index Performance - 3 Year Average")

index_performance = df_filtered.groupby('Index')['Positive'].mean().sort_values(ascending=False).reset_index()
index_performance['Color'] = index_performance['Positive'].apply(
    lambda v: color_map['Strength'] if v >= strength_threshold else color_map['Neutral'] if v >= weakness_threshold else color_map['Weakness']
)

fig_idx = px.bar(
    index_performance, x='Positive', y='Index', orientation='h',
    color='Color', color_discrete_map='identity',
    labels={'Positive': '% Positive'}, height=300 + (len(index_performance) * 10)
)

fig_idx.add_vline(x=strength_threshold, line_dash="dash", line_color="green",
                  annotation_text=f"Strength: {strength_threshold}%", annotation_position="top right")
fig_idx.add_vline(x=weakness_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Weakness: {weakness_threshold}%", annotation_position="top left")
fig_idx.update_traces(text=index_performance['Positive'].round(1).astype(str)+'%', textposition='inside')
fig_idx.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_idx, use_container_width=True)


# Sub-index breakdown (hide if only one sub-index)
# st.markdown("### Index & Sub-Index Breakdown - 3 Years Average")
index_breakdown = df_filtered.groupby(['Index', 'Sub.Index'])['Positive'].mean().reset_index()
sub_counts = index_breakdown.groupby('Index')['Sub.Index'].nunique()
# if selected index has only one sub-index, hide the large bar
if selected_index == 'All' or sub_counts.get(selected_index, 0) > 1:
    index_avg = df_filtered.groupby('Index')['Positive'].mean().sort_values(ascending=False)
    index_breakdown['Index'] = pd.Categorical(index_breakdown['Index'], categories=index_avg.index)
    index_breakdown = index_breakdown.sort_values(['Index', 'Positive'], ascending=[False, False])
    st.markdown("### Index & Sub-Index Breakdown - 3 Years Average")
    fig3 = px.bar(index_breakdown, x='Positive', y='Sub.Index', color='Index',
                  orientation='h', labels={'Positive': '% Positive', 'Sub.Index': 'Sub-Index'},
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig3.add_vline(x=strength_threshold, line_dash="dash", line_color="green", annotation_text=f"Strength: {strength_threshold}%", annotation_position="top right")
    fig3.add_vline(x=weakness_threshold, line_dash="dash", line_color="red", annotation_text=f"Weakness: {weakness_threshold}%", annotation_position="top left")
    fig3.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig3, use_container_width=True)

if selected_index == 'All':
    st.markdown("#### Distribution of question average scores (current view)")
    # add cutoff lines for strength/weakness thresholds
    # color the regions to the left of the weakness threshold and right of the strength threshold
    hist = px.histogram(pf.reset_index(), x='Avg', nbins=12, labels={'Avg': 'Average % Positive'})
    hist.add_vline(x=strength_threshold, line_dash="dash", line_color="green",
                   annotation_text=f"Strength threshold: {strength_threshold}%", annotation_position="top right")
    hist.add_vline(x=weakness_threshold, line_dash="dash", line_color="red",
                   annotation_text=f"Weakness threshold: {weakness_threshold}%", annotation_position="top left")
    hist.add_shape(type="rect",x0=pf['Avg'].min(),x1=weakness_threshold,y0=0,y1=1,xref="x",yref="paper",fillcolor="rgba(255, 0, 0, 0.15)", line_width=0, layer="below" )
    hist.add_shape(type="rect",x0=strength_threshold,x1=pf['Avg'].max(),y0=0,y1=1,xref="x",yref="paper",fillcolor="rgba(0, 200, 0, 0.15)", line_width=0, layer="below" )
    hist.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(hist, use_container_width=True)

# ---------------------------
# Question Analysis tab (drilldown + heatmap)
# ---------------------------
st.markdown("### Question performance heatmap")

# Heatmap data with question label
pivot_display = pivot_filtered[['2023', '2024', '2025']].sort_values(by='2025', ascending=False)
pivot_with_text = pivot_display.reset_index().merge(q_map, on='Question')
pivot_with_text['Display'] = pivot_with_text['Question'] + ': ' + pivot_with_text['Item.Text'].apply(
    lambda x: truncate_text(x, 30))

# Create heatmap with hover text
fig6 = go.Figure(data=go.Heatmap(
    z=pivot_display.values,
    x=['2023', '2024', '2025'],  # Use strings for categorical
    y=pivot_with_text['Display'],
    colorscale='RdYlGn',
    text=pivot_display.values.round(1),
    texttemplate='%{text}%',
    hovertemplate='Year: %{x}<br>%{y}<br>Positive: %{z:.1f}%<extra></extra>',
    colorbar=dict(title='% Positive')
))

# Compute the average score for each question across the heatmap years
row_avgs = pivot_display.mean(axis=1)
for row_i, avg in enumerate(row_avgs):
    if avg >= strength_threshold:
        # Highlight the entire row in green
        fig6.add_shape(
            type="rect",
            x0=-0.5, x1=len(pivot_display.columns) - 0.5,
            y0=row_i - 0.5, y1=row_i + 0.5,
            xref="x", yref="y",
            line=dict(color="green", width=2)
        )
    elif avg <= weakness_threshold:
        # Highlight the entire row in red
        fig6.add_shape(
            type="rect",
            x0=-0.5, x1=len(pivot_display.columns) - 0.5,
            y0=row_i - 0.5, y1=row_i + 0.5,
            xref="x", yref="y",
            line=dict(color="red", width=2)
        )

fig6.update_layout(height=max(400, len(pivot_display) * 20),
                   margin=dict(l=0, r=0, t=10, b=0),
                   xaxis_title="", yaxis_title="")
fig6.update_xaxes(type='category')  # Force categorical axis
st.plotly_chart(fig6, use_container_width=True)

# Question drilldown (dynamic)
if selected_question and selected_question != 'None':
    st.markdown("---")
    st.markdown(f"#### Question detail: {selected_question}")
    q_info = q_map[q_map['Question'] == selected_question].iloc[0]
    q_data = df_results[df_results['Question'] == selected_question].copy()
    q_data['Year'] = q_data['Year'].astype(str)

    st.markdown(f"**{q_info['Item.Text']}**")
    st.caption(f"{q_info['Index']} › {q_info['Sub.Index']}")

    c1, c2, c3, c4 = st.columns(4)
    avg_pos = q_data['Positive'].mean()
    trend = pivot.loc[selected_question, '2025'] - pivot.loc[
        selected_question, '2023'] if '2023' in pivot.columns and '2025' in pivot.columns else np.nan
    c1.metric("Avg", f"{avg_pos:.1f}%")
    c2.metric("2023", f"{pivot.loc[selected_question, '2023']:.1f}%")
    c3.metric("2024", f"{pivot.loc[selected_question, '2024']:.1f}%")
    c4.metric("2025", f"{pivot.loc[selected_question, '2025']:.1f}%")

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Positive'], mode='lines+markers', name='Positive',
                              line=dict(color='green', width=3), marker=dict(size=10)))
    fig7.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Neutral'], mode='lines+markers', name='Neutral',
                              line=dict(color='gray', width=2), marker=dict(size=8)))
    fig7.add_trace(go.Scatter(x=q_data['Year'], y=q_data['Negative'], mode='lines+markers', name='Negative',
                              line=dict(color='red', width=2), marker=dict(size=8)))
    fig7.add_hline(y=strength_threshold, line_dash="dash", line_color="green",
                   annotation_text=f"Strength threshold: {strength_threshold}%", annotation_position="top right")
    fig7.add_hline(y=weakness_threshold, line_dash="dash", line_color="red",
                   annotation_text=f"Weakness threshold: {weakness_threshold}%", annotation_position="bottom right")
    fig7.update_layout(hovermode='x unified', height=300, yaxis_title="Percentage", margin=dict(l=0, r=0, t=10, b=0))
    fig7.update_xaxes(type='category')
    st.plotly_chart(fig7, use_container_width=True)