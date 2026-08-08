import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Page Configuration
st.set_page_config(
    page_title="Palo Alto Networks - Attrition Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# Load and Preprocess Dataset
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('Palo Alto Networks.csv')
    if df['Attrition'].dtype == object:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        
    df['Income_Per_Year'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['Promotion_Delay_Ratio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['Engagement_Score'] = (
        df['EnvironmentSatisfaction'] + df['JobInvolvement'] + 
        df['JobSatisfaction'] + df['RelationshipSatisfaction']
    ) / 4.0
    df['Workload_Stress_Flag'] = np.where(
        (df['OverTime'] == 'Yes') & (df['WorkLifeBalance'] <= 2), 1, 0
    )
    return df

# Train and Cache Model in Memory (No pickle/joblib required)
@st.cache_resource
def train_model(df):
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
        ]
    )

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            eval_metric='logloss',
            random_state=42
        ))
    ])

    pipeline.fit(X, y)
    return pipeline

# Load Data and Train Model
df = load_and_preprocess_data()

with st.spinner("Initializing Predictive Engine..."):
    model = train_model(df)

# Inference Generation
X_feat = df.drop(columns=['Attrition'], errors='ignore')
df['Attrition_Probability'] = model.predict_proba(X_feat)[:, 1]

df['Risk_Category'] = pd.cut(
    df['Attrition_Probability'],
    bins=[-0.01, 0.30, 0.60, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Header
st.title("🛡️ Employee Attrition Prediction & Risk Scoring System")
st.caption("Palo Alto Networks - ML-Driven Workforce Retention Portal")

# Sidebar Filter
st.sidebar.header("Filter Analytics")
dept_filter = st.sidebar.multiselect(
    "Select Department:",
    options=df['Department'].unique(),
    default=df['Department'].unique()
)

filtered_df = df[df['Department'].isin(dept_filter)]

# Tabs Interface
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", 
    "👤 Employee Risk Profile", 
    "🏢 Department Risk Breakdown", 
    "🔮 What-If Simulator"
])

# --- TAB 1: EXECUTIVE SUMMARY ---
with tab1:
    st.subheader("Workforce Attrition Risk Metrics")
    c1, c2, c3, c4 = st.columns(4)
    
    tot_emp = len(filtered_df)
    high_risk = len(filtered_df[filtered_df['Risk_Category'] == 'High Risk'])
    med_risk = len(filtered_df[filtered_df['Risk_Category'] == 'Medium Risk'])
    avg_score = filtered_df['Attrition_Probability'].mean() * 100
    
    c1.metric("Total Workforce Filtered", tot_emp)
    c2.metric("High-Risk Count (>60%)", high_risk, delta=f"{high_risk/tot_emp*100:.1f}%", delta_color="inverse")
    c3.metric("Medium-Risk Count (30-60%)", med_risk)
    c4.metric("Avg Risk Score", f"{avg_score:.1f}%")
    
    st.markdown("---")
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_pie = px.pie(
            filtered_df, names='Risk_Category', title="Risk Category Distribution",
            color='Risk_Category',
            color_discrete_map={'Low Risk':'#2ecc71', 'Medium Risk':'#f39c12', 'High Risk':'#e74c3c'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b:
        fig_box = px.box(
            filtered_df, x='OverTime', y='Attrition_Probability', color='OverTime',
            title="OverTime Exposure vs. Attrition Probability"
        )
        st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 2: EMPLOYEE RISK PROFILE ---
with tab2:
    st.subheader("Individual Risk Drilldown")
    emp_idx = st.selectbox("Select Employee Record Index:", filtered_df.index)
    emp_data = filtered_df.loc[emp_idx]
    
    col_x, col_y = st.columns([1, 2])
    with col_x:
        st.markdown("#### Employee Information")
        st.write(f"**Department:** {emp_data['Department']}")
        st.write(f"**Job Role:** {emp_data['JobRole']}")
        st.write(f"**Age:** {emp_data['Age']}")
        st.write(f"**Monthly Income:** ${emp_data['MonthlyIncome']:,}")
        st.write(f"**OverTime:** {emp_data['OverTime']}")
        st.write(f"**Years At Company:** {emp_data['YearsAtCompany']}")
        st.write(f"**Distance From Home:** {emp_data['DistanceFromHome']} km")
        
    with col_y:
        st.markdown("#### Risk Score")
        prob = emp_data['Attrition_Probability']
        st.progress(float(prob))
        st.markdown(f"### Score: `{prob*100:.1f}%`  |  Category: **{emp_data['Risk_Category']}**")
        
        st.markdown("#### Primary Driver Indicators")
        if emp_data['OverTime'] == 'Yes':
            st.warning("⚠️ **High OverTime Exposure:** Frequent overtime contributes to elevated exit risk.")
        if emp_data['WorkLifeBalance'] <= 2:
            st.warning("⚠️ **Poor Work-Life Balance:** Rating is 2 or lower.")
        if emp_data['YearsSinceLastPromotion'] >= 4:
            st.warning(f"⚠️ **Promotion Stagnation:** No promotion in {emp_data['YearsSinceLastPromotion']} years.")

# --- TAB 3: DEPARTMENT BREAKDOWN ---
with tab3:
    st.subheader("Department & Role Breakdown")
    dept_chart = px.histogram(
        filtered_df, x="Department", color="Risk_Category", barmode="group",
        color_discrete_map={'Low Risk':'#2ecc71', 'Medium Risk':'#f39c12', 'High Risk':'#e74c3c'},
        title="Risk Tiers Across Departments"
    )
    st.plotly_chart(dept_chart, use_container_width=True)

# --- TAB 4: WHAT-IF SIMULATOR ---
with tab4:
    st.subheader("Interactive Retention Policy Simulator")
    st.markdown("Simulate how policy changes alter an employee's attrition probability.")
    
    emp_sim_idx = st.selectbox("Select Employee to Simulate:", filtered_df.index, key="sim_select")
    emp_sim = filtered_df.loc[emp_sim_idx]
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.markdown("**Simulate Interventions:**")
        hike = st.slider("Salary Increase (%)", 0, 30, 0, step=5)
        remove_ot = st.checkbox("Remove OverTime Requirement")
        
    with s_col2:
        orig_p = emp_sim['Attrition_Probability']
        sim_p = orig_p
        
        if hike > 0:
            sim_p *= (1 - (hike / 100) * 0.75)
        if remove_ot and emp_sim['OverTime'] == 'Yes':
            sim_p *= 0.60
            
        st.metric("Baseline Attrition Score", f"{orig_p*100:.1f}%")
        st.metric(
            "Simulated Attrition Score", f"{sim_p*100:.1f}%", 
            delta=f"{(sim_p - orig_p)*100:.1f}%", delta_color="inverse"
        )
