import streamlit as st
import pandas as pd
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ReportLab for PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---------------------------
# Import XGBoost explicitly
# ---------------------------
try:
    import xgboost
except ImportError:
    st.error("❌ XGBoost is not installed. Please run: pip install xgboost")
    st.stop()

# ---------------------------
# Streamlit page config & header
# ---------------------------
st.set_page_config(page_title="CAD Prediction App", page_icon="🫀", layout="wide")
st.markdown("""
<div style='text-align: center;'>
    <h1>Coronary Artery Disease Prediction using Gut Microbiome</h1>
    <p>Explore how gut microbiome composition can help predict CAD risk</p>
</div>
""", unsafe_allow_html=True)
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

            if "Status" not in data.columns:
                st.error("Dataset must contain a 'Status' column with 'cad' or 'control'")
            else:
                # Load model & features
                model = pickle.load(open("cad_model.pkl", "rb"))
                with open("cad_features.json", "r") as f:
                    expected_features = json.load(f)

                X = data.drop(columns=["Status"], errors="ignore")
                y_true = data["Status"].str.lower() if "Status" in data.columns else None

                # Encode categorical
                for col in X.select_dtypes(include="object").columns:
                    X[col] = pd.factorize(X[col])[0]

                # Add missing cols / drop extras
                for c in expected_features:
                    if c not in X.columns:
                        X[c] = 0
                X = X[expected_features]

                # ---------------------------
                # Predict CAD
                # ---------------------------
                if st.button("Predict CAD"):
                    preds = model.predict(X)
                    mapping = {0: "cad", 1: "control"}
                    data["Predicted_Status"] = [mapping.get(p, p) for p in preds]
                    if y_true is not None:
                        data["Actual_Status"] = y_true

                    st.success("✅ Predictions completed!")
                    st.dataframe(data.head())

                    # Prediction Ratio
                    pred_counts = data["Predicted_Status"].value_counts()
                    fig1, ax1 = plt.subplots()
                    sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax1, palette="Set2")
                    ax1.set_title("CAD vs Control Predictions")
                    ax1.set_ylabel("Count")
                    ax1.set_xlabel("Predicted Status")
                    st.pyplot(fig1)

                    # Confusion Matrix & Accuracy
                    if y_true is not None:
                        cm = confusion_matrix(y_true, data["Predicted_Status"], labels=["cad", "control"])
                        fig2, ax2 = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                    xticklabels=["cad", "control"],
                                    yticklabels=["cad", "control"],
                                    ax=ax2)
                        ax2.set_xlabel("Predicted")
                        ax2.set_ylabel("Actual")
                        st.pyplot(fig2)

                        report = classification_report(y_true, data["Predicted_Status"], output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.background_gradient(cmap="Blues"))

                        acc = accuracy_score(y_true, data["Predicted_Status"])
                        st.write(f"**Accuracy:** {acc:.2f}")
                    else:
                        acc = None
                        fig2 = None

                    # Feature Importance
                    if hasattr(model, "feature_importances_"):
                        importance = model.feature_importances_
                        feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importance})
                        feat_imp = feat_imp.sort_values(by="Importance", ascending=False)
                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        sns.barplot(x="Importance", y="Feature", data=feat_imp, ax=ax3, palette="viridis")
                        ax3.set_title("Feature Importance")
                        st.pyplot(fig3)
                    else:
                        fig3 = None

                    # Save in session_state
                    st.session_state["fig1"] = fig1
                    st.session_state["fig2"] = fig2
                    st.session_state["fig3"] = fig3
                    st.session_state["acc"] = acc
                    st.session_state["data"] = data.copy()

                # ---------------------------
                # Generate PDF using session_state
                # ---------------------------
                if st.button("📄 Generate PDF Report"):
                    if "fig1" not in st.session_state:
                        st.error("Please run 'Predict CAD' first!")
                    else:
                        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        pdf_file = tmp_pdf.name
                        tmp_pdf.close()

                        doc = SimpleDocTemplate(pdf_file)
                        styles = getSampleStyleSheet()
                        story = []

                        # Title
                        story.append(Paragraph("CAD Prediction Report", styles["Title"]))
                        story.append(Spacer(1, 20))

                        # Accuracy
                        if st.session_state["acc"] is not None:
                            story.append(Paragraph(f"Accuracy: {st.session_state['acc']:.2f}", styles["Normal"]))
                            story.append(Spacer(1, 10))

                        # Classification report table
                        if "data" in st.session_state and st.session_state["data"] is not None:
                            report = classification_report(
                                st.session_state["data"]["Actual_Status"],
                                st.session_state["data"]["Predicted_Status"],
                                output_dict=True
                            )
                            report_df = pd.DataFrame(report).transpose().round(2)

                            table_data = [["Class", "Precision", "Recall", "F1-Score", "Support"]]
                            for cls in report_df.index:
                                if cls == "accuracy":
                                    continue
                                row = [
                                    cls,
                                    report_df.loc[cls, "precision"],
                                    report_df.loc[cls, "recall"],
                                    report_df.loc[cls, "f1-score"],
                                    int(report_df.loc[cls, "support"])
                                ]
                                table_data.append(row)

                            tbl = Table(table_data, hAlign='LEFT')
                            tbl.setStyle(TableStyle([
                                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                                ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                                ('ALIGN',(1,1),(-1,-1),'CENTER'),
                                ('GRID', (0,0), (-1,-1), 1, colors.black),
                                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                            ]))
                            story.append(Paragraph("Classification Report", styles["Heading2"]))
                            story.append(tbl)
                            story.append(Spacer(1, 20))

                        # Add plots
                        for fig, title in zip(
                            [st.session_state["fig1"], st.session_state["fig2"], st.session_state["fig3"]],
                            ["Prediction Ratio", "Confusion Matrix", "Feature Importance"]
                        ):
                            if fig is not None:
                                tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                fig.savefig(tmp_img.name, bbox_inches="tight", facecolor="white")
                                story.append(Paragraph(title, styles["Heading2"]))
                                story.append(Image(tmp_img.name, width=400, height=250))
                                story.append(Spacer(1, 20))
                                tmp_img.close()

                        # Build PDF
                        doc.build(story)

                        # Streamlit download button
                        with open(pdf_file, "rb") as f:
                            st.download_button(
                                label="📥 Download PDF",
                                data=f,
                                file_name="CAD_Prediction_Report.pdf",
                                mime="application/pdf"
                            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
