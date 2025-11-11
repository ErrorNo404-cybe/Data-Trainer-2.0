import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix
)
import joblib
import io
import base64
import zipfile
from io import BytesIO

# Page Configuration
st.set_page_config(
    page_title="Avash's Data Training Platform 2.0",
    page_icon="🚀",
    layout="wide"
)

# Title & Intro
st.title("🚀 Avash's Data Training Platform 2.0")
st.markdown(
    """
    End-to-end machine learning without writing a single line of code.  
    From business problem → trained model → live deployment — all in one place!
    """
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# STEP 1: Business Understanding
st.header("1️⃣ Business Understanding & Problem Formulation")
problem_type = st.selectbox(
    "Select your problem type:",
    ["Classification", "Regression"],
    help="Classification: predict categories | Regression: predict numbers"
)
business_goal = st.text_area(
    "Briefly describe your business goal (optional):",
    placeholder="e.g., Predict customer churn..."
)

# STEP 2: Data Acquisition
st.header("2️⃣ Data Acquisition and Collection")
col_upload, col_sample = st.columns([2, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])
with col_sample:
    st.markdown("### 💡 Try a sample:")
    if st.button("Iris (Classification)"):
        from sklearn.datasets import load_iris
        data = load_iris()
        df_sample = pd.DataFrame(data.data, columns=data.feature_names)
        df_sample["species"] = data.target
        st.session_state.df = df_sample
        st.rerun()
    if st.button("Diabetes (Regression)"):
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df_sample = pd.DataFrame(data.data, columns=data.feature_names)
        df_sample["target"] = data.target
        st.session_state.df = df_sample
        st.rerun()

# Load uploaded data
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.df = df
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

df = st.session_state.df

if df is not None:
    st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    with st.expander("🔍 View Dataset"):
        st.dataframe(df)

    # STEP 3: Data Cleaning
    st.header("3️⃣ Data Cleaning and Preparation")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning(f"⚠️ Missing values in: {missing[missing > 0].index.tolist()}")
        strategy = st.radio("Handle missing values:", 
                           ["Drop rows", "Impute (median/mode)"])
        if strategy == "Drop rows":
            df = df.dropna()
        else:
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
    
    # Duplicates
    dup = df.duplicated().sum()
    if dup > 0 and st.checkbox(f"Remove {dup} duplicate(s)?"):
        df = df.drop_duplicates()
    
    st.session_state.cleaned_df = df

    # STEP 4: EDA
    st.header("4️⃣ Exploratory Data Analysis (EDA)")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    if st.checkbox("📊 Summary Statistics"):
        st.write(df.describe(include='all'))
    
    if num_cols:
        st.subheader("📈 Distribution")
        col = st.selectbox("Select feature", num_cols)
        st.plotly_chart(px.histogram(df, x=col), use_container_width=True)
    
    if len(num_cols) >= 2:
        st.subheader("🔍 Scatter Plot")
        x, y = st.columns(2)
        with x: x_axis = st.selectbox("X", num_cols, key="x")
        with y: y_axis = st.selectbox("Y", num_cols, key="y")
        color = st.selectbox("Color (optional)", [None] + cat_cols, key="c")
        st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis, color=color), use_container_width=True)
    
    if len(num_cols) > 1:
        st.subheader("🧮 Correlation")
        st.plotly_chart(px.imshow(df[num_cols].corr(), text_auto=True), use_container_width=True)

    # STEP 5-6: Modeling & Evaluation
    st.header("5️⃣ Model Building & Evaluation")
    target = st.selectbox("🎯 Select Target", df.columns)
    
    if target:
        # Show target info
        st.subheader("📊 Target Info")
        st.write(f"Type: `{df[target].dtype}`, Unique: {df[target].nunique()}")
        if df[target].nunique() < 20:
            st.write(f"Values: {sorted(df[target].unique())}")
        
        # Validate problem type vs target
        if problem_type == "Classification":
            if df[target].nunique() > 20 and df[target].dtype != 'object':
                st.error("❌ Classification requires a categorical/discrete target. Switch to **Regression** or choose a different target.")
                st.stop()
        elif problem_type == "Regression":
            if df[target].dtype == 'object' or df[target].nunique() < 3:
                st.warning("⚠️ Regression usually uses continuous targets. Is this really not a classification problem?")
        
        X = df.drop(columns=[target])
        y = df[target]
        
        # Encode features
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if needed
        le_target = None
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Safe train-test split
        if problem_type == "Classification":
            class_counts = pd.Series(y).value_counts()
            if (class_counts < 2).any():
                st.warning(f"⚠️ Small classes detected: {class_counts[class_counts < 2].to_dict()}")
                st.info("Disabling stratification.")
                stratify = None
            else:
                stratify = y
        else:
            stratify = None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        
        if st.button("🚀 Train & Evaluate Model"):
            with st.spinner("Training..."):
                if problem_type == "Classification":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='weighted')
                    rec = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{acc:.3f}")
                    col2.metric("Precision", f"{prec:.3f}")
                    col3.metric("Recall", f"{rec:.3f}")
                    col4.metric("F1", f"{f1:.3f}")
                    
                    st.subheader("📉 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    st.plotly_chart(px.imshow(cm, text_auto=True), use_container_width=True)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("MSE", f"{mse:.3f}")
                    col2.metric("R²", f"{r2:.3f}")
                    
                    st.subheader("📉 Actual vs Predicted")
                    fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"})
                    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode="lines", name="Ideal"))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                st.subheader("🔑 Feature Importance")
                imp_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
                st.plotly_chart(px.bar(imp_df, x="Importance", y="Feature", orientation="h"), use_container_width=True)
                
                # Save model
                st.session_state.model = model
                st.session_state.feature_names = list(X.columns)
                st.session_state.le_target = le_target
                st.session_state.problem_type = problem_type
                
                # Raw model download
                buf = io.BytesIO()
                joblib.dump(model, buf)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                st.markdown(f'<a href="file/pkl;base64,{b64}" download="model.pkl">📥 Download Model (.pkl)</a>', unsafe_allow_html=True)

                # STEP 7: Deployment Bundle
                st.header("6️⃣ Deployment and Monitoring")
                if st.button("📦 Generate Deployable App (Gradio + Hugging Face)"):
                    with st.spinner("Generating app..."):
                        feature_names_clean = [f.replace(" ", "_").replace("-", "_") for f in X.columns]
                        feature_args = ", ".join(feature_names_clean)
                        
                        if problem_type == "Classification":
                            if le_target is not None:
                                labels = {i: str(cls) for i, cls in enumerate(le_target.classes_)}
                                label_code = f"LABELS = {labels}\n"
                                predict_code = f"""    pred = model.predict([[{feature_args}]])[0]
    return LABELS.get(int(pred), str(pred))"""
                            else:
                                label_code = ""
                                predict_code = f"""    return str(model.predict([[{feature_args}]])[0])"""
                            
                            gradio_code = f'''import gradio as gr
import joblib
model = joblib.load("model.pkl")
{label_code}
def predict({feature_args}):
{predict_code}
inputs = [gr.Number(label="{name}") for name in {list(X.columns)}]
gr.Interface(fn=predict, inputs=inputs, outputs="text", title="Predictor").launch()
'''
                        else:
                            gradio_code = f'''import gradio as gr
import joblib
model = joblib.load("model.pkl")
def predict({feature_args}):
    return float(model.predict([[{feature_args}]])[0])
inputs = [gr.Number(label="{name}") for name in {list(X.columns)}]
gr.Interface(fn=predict, inputs=inputs, outputs="number", title="Predictor").launch()
'''
                        
                        # Create ZIP
                        zip_buf = BytesIO()
                        with zipfile.ZipFile(zip_buf, "w") as zf:
                            model_io = BytesIO()
                            joblib.dump(model, model_io)
                            model_io.seek(0)
                            zf.writestr("model.pkl", model_io.read())
                            zf.writestr("app.py", gradio_code.strip())
                            zf.writestr("requirements.txt", "gradio\nscikit-learn\npandas\nnumpy\njoblib")
                            zf.writestr("README.md", """# Deploy to Hugging Face Spaces

1. Go to https://huggingface.co/spaces
2. Create new Space → SDK: Gradio
3. Upload this ZIP
4. Done!
""")
                        
                        zip_buf.seek(0)
                        st.download_button(
                            "⬇️ Download Deployable App (ZIP)",
                            zip_buf,
                            "deployable_model_app.zip",
                            "application/zip"
                        )
                        st.success("✅ Upload ZIP to Hugging Face Spaces to go live!")

else:
    st.info("👆 Upload a dataset or use a sample to begin!")

st.markdown("---")
st.caption("💡 Built with Streamlit • Models: Random Forest • Deployment: Hugging Face Spaces")
st.caption("© 2025 Avash's Data Trainer Platform. All rights reserved.")
st.caption(
    "🔗 Previous version available at: [avash-data-trainer.streamlit.app](https://avash-data-trainer.streamlit.app)"
)