import streamlit as st
import pandas as pd
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from fpdf import FPDF

# ---------------------------
# Import XGBoost explicitly
# ---------------------------
try:
    import xgboost
except ImportError:
    st.error("❌ XGBoost is not installed. Please run: pip install xgboost")
    st.stop()

# ---------------------------
# Page Header
# ---------------------------
st.set_page_config(page_title="CAD Prediction App", page_icon="🫀", layout="wide")
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Coronary Artery Disease Prediction using Gut Microbiome</h1>
        <p>Explore how gut microbiome composition can help predict CAD risk</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "CAD Prediction Tool"])

# ---------------------------
# Page 1: Home
# ---------------------------
if page == "Home":
    st.subheader("Introduction")
    st.write("""
    Coronary Artery Disease (CAD) is a leading cause of death worldwide.  
    Gut microbiome composition can help predict CAD risk.
    """)
    img_path = os.path.join(os.path.dirname(__file__), "Main.jpg")
    if os.path.exists(img_path):
        st.image(img_path, caption="Gut Microbiome & CAD", use_container_width=True)
    else:
        st.warning("Image file 'Main.jpg' not found.")

# ---------------------------
# Page 2: CAD Prediction Tool
# ---------------------------
elif page == "CAD Prediction Tool":
    st.subheader("CAD Prediction Tool")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load dataset
            data = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview")
            st.dataframe(data.head())

            # Check target column
            if "Status" not in data.columns:
                st.error("Dataset must contain a 'Status' column with 'cad' or 'control'")
            else:
                # Load model
                model = pickle.load(open("cad_model.pkl", "rb"))

                # Load expected feature names
                with open("cad_features.json", "r") as f:
                    expected_features = json.load(f)

                # Prepare features
                X = data.drop(columns=["Status"], errors="ignore")
                y_true = data["Status"].str.lower() if "Status" in data.columns else None

                # Encode categorical columns (like clade_name)
                for col in X.select_dtypes(include="object").columns:
                    X[col] = pd.factorize(X[col])[0]

                # Add missing columns, drop extras
                for c in expected_features:
                    if c not in X.columns:
                        X[c] = 0
                X = X[expected_features]

                # Predict button
                if st.button("Predict CAD"):
                    preds = model.predict(X)
                    mapping = {0: "cad", 1: "control"}
                    data["Predicted_Status"] = [mapping.get(p, p) for p in preds]

                    if y_true is not None:
                        data["Actual_Status"] = y_true

                    st.success("✅ Predictions completed!")
                    st.dataframe(data.head())

                    # ---------------------------
                    # Prediction Ratio Plot
                    # ---------------------------
                    st.write("### Prediction Ratio")
                    pred_counts = data["Predicted_Status"].value_counts()
                    fig1, ax1 = plt.subplots()
                    sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax1, palette="Set2")
                    ax1.set_title("CAD vs Control Predictions")
                    ax1.set_ylabel("Count")
                    ax1.set_xlabel("Predicted Status")
                    st.pyplot(fig1)

                    # ---------------------------
                    # Confusion Matrix
                    # ---------------------------
                    if y_true is not None:
                        st.write("### Confusion Matrix")
                        cm = confusion_matrix(y_true, data["Predicted_Status"], labels=["cad", "control"])
                        fig2, ax2 = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                    xticklabels=["cad", "control"],
                                    yticklabels=["cad", "control"],
                                    ax=ax2)
                        ax2.set_xlabel("Predicted")
                        ax2.set_ylabel("Actual")
                        st.pyplot(fig2)

                        # Classification Report
                        st.write("### Classification Report")
                        report = classification_report(y_true, data["Predicted_Status"], output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.background_gradient(cmap="Blues"))

                        acc = accuracy_score(y_true, data["Predicted_Status"])
                        st.write(f"**Accuracy:** {acc:.2f}")

                    # ---------------------------
                    # Feature Importance
                    # ---------------------------
                    if hasattr(model, "feature_importances_"):
                        st.write("### Feature Importance")
                        importance = model.feature_importances_
                        feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importance})
                        feat_imp = feat_imp.sort_values(by="Importance", ascending=False)
                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        sns.barplot(x="Importance", y="Feature", data=feat_imp, ax=ax3, palette="viridis")
                        ax3.set_title("Feature Importance")
                        st.pyplot(fig3)

                    # ---------------------------
                    # Generate PDF Report
                    # ---------------------------
                    if st.button("📄 Generate PDF Report"):
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", "B", 16)
                        pdf.cell(0, 10, "CAD Prediction Report", ln=True, align="C")
                        pdf.ln(10)

                        # Add accuracy
                        if y_true is not None:
                            pdf.set_font("Arial", "", 12)
                            pdf.cell(0, 10, f"Accuracy: {acc:.2f}", ln=True)
                            pdf.ln(5)

                        # Save plots to temp files and add
                        for fig, title in zip([fig1, fig2, fig3],
                                              ["Prediction Ratio", "Confusion Matrix", "Feature Importance"]):
                            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            fig.savefig(tmpfile.name, bbox_inches="tight")
                            pdf.ln(5)
                            pdf.cell(0, 10, title, ln=True)
                            pdf.image(tmpfile.name, x=10, w=190)
                            tmpfile.close()

                        pdf_file = "CAD_Prediction_Report.pdf"
                        pdf.output(pdf_file)
                        st.success(f"✅ PDF Report generated: {pdf_file}")
                        with open(pdf_file, "rb") as f:
                            st.download_button("📥 Download PDF", f, file_name=pdf_file)

        except Exception as e:
            st.error(f"Error processing file: {e}")
