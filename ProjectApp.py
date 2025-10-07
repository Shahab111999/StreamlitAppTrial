import streamlit as st
import pandas as pd
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import joblib
import sys
# Top of your script
try:
    from sklearn.compose._column_transformer import _RemainderColsList
except ImportError:
    class _RemainderColsList(list):
        """Dummy class to allow loading old pickled models."""
        pass

# Add it to the module namespace so pickle can find it
sys.modules['sklearn.compose._column_transformer']._RemainderColsList = _RemainderColsList

def cast_to_str(A):
    try:
        return A.astype(str)
    except Exception:
        import numpy as np, pandas as pd
        if isinstance(A, pd.DataFrame):
            return A.apply(lambda col: col.astype(str))
        return np.asarray(A).astype(str)
    
    
# --- Cached model/feature loaders ---
@st.cache_resource
def load_model(model_name="xgboost"):
    
    if model_name == "xgboost":
        with open("xgboost_filtered_data_model.pkl", "rb") as f:
            xgb_model = joblib.load(f)
        return xgb_model
    else:
        with open("models/svm_filtered_data_model.pkl", "rb") as f:
            xgb_model = joblib.load(f)
        return xgb_model


@st.cache_data
def load_feature_names():
    with open("feature_names_in_order_filtered_data.json", "r") as f:
        feature_names = json.load(f)
        print(f"Loaded {len(feature_names)} feature names.")
    if not isinstance(feature_names, (list, tuple)):
        raise ValueError("feature_names_in_order must be a JSON array")
    return feature_names



# ---------------------------
# Helper function for Japan predictions
# ---------------------------
def run_japan_prediction(data, model_file, features_file, label_col, dataset_name):
    # --- You can copy your existing run_prediction logic here ---
    st.write(f"Japan Prediction Page for {dataset_name}")
    st.dataframe(data.head())
    st.info("This page uses the existing Japan prediction code.")
    # You can include prediction, plots, PDF export here as in your current code

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="CAD Prediction App", page_icon="🫀", layout="wide")
st.sidebar.title("🫀 CAD Prediction App")

page = st.sidebar.radio(
    "Navigate to:",
    ["Home", "Japan Gut Prediction", "USA Custom Page", "Datasets", "About"]
)

# ---------------------------
# Home Page
# ---------------------------
if page == "Home":
    st.markdown("<h1 style='text-align:center;'>🫀 CAD Prediction using Gut Microbiome</h1>", unsafe_allow_html=True)
    st.write("Welcome! Use the sidebar to navigate to Japan or USA dataset pages.")

# ---------------------------
# Japan Gut Prediction Page
# ---------------------------
# ---------------------------
# Page: Japan Gut Prediction
# ---------------------------
elif page == "Japan Gut Prediction":
    st.markdown("## 🇯🇵 Japan Gut Dataset Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload Japan dataset (CSV) with column `Status`", 
        type="csv", 
        key="japan_file"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview")
            st.dataframe(data.head())

            if "Status" not in data.columns:
                st.error("⚠️ Dataset must contain a 'Status' column with 'cad' or 'control'")
            else:
                # Load Japan model and features
                model_file = "best_model_japan.pkl"
                features_file = "best_model_japan.json"
                
                model = pickle.load(open(model_file, "rb"))
                with open(features_file, "r") as f:
                    expected_features = json.load(f)

                X = data.drop(columns=["Status"], errors="ignore")
                y_true = data["Status"].astype(str).str.lower()

                # Factorize categorical columns
                for col in X.select_dtypes(include="object").columns:
                    X[col] = pd.factorize(X[col])[0]

                # Fill missing columns with zeros
                missing_cols = [c for c in expected_features if c not in X.columns]
                if missing_cols:
                    st.warning(f"⚠️ Missing columns filled with zeros: {missing_cols}")
                    for c in missing_cols:
                        X[c] = 0

                # Reorder columns
                X = X[expected_features]

                # Run Prediction
                if st.button("🚀 Run Japan Gut Prediction"):
                    preds = model.predict(X)
                    mapping = {0: "cad", 1: "control"}
                    data["Predicted_Status"] = [mapping.get(p, p) for p in preds]
                    data["Actual_Status"] = y_true

                    st.success("✅ Predictions completed!")
                    st.dataframe(data.head())

                    # Prediction distribution plot
                    pred_counts = data["Predicted_Status"].value_counts()
                    fig1, ax1 = plt.subplots()
                    sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax1, palette="Set2")
                    ax1.set_title("Japan Gut Prediction Distribution")
                    st.pyplot(fig1)

                    # Confusion matrix plot
                    cm = confusion_matrix(y_true, data["Predicted_Status"], labels=["cad", "control"])
                    fig2, ax2 = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=["cad", "control"], yticklabels=["cad", "control"],
                                ax=ax2)
                    ax2.set_xlabel("Predicted")
                    ax2.set_ylabel("Actual")
                    st.pyplot(fig2)

                    # Accuracy
                    acc = accuracy_score(y_true, data["Predicted_Status"])
                    st.markdown(f"### 🎯 Model Accuracy: **{acc:.2f}**")

                    # Save session for PDF report
                    st.session_state["japan_data"] = data.copy()
                    st.session_state["japan_fig1"] = fig1
                    st.session_state["japan_fig2"] = fig2
                    st.session_state["japan_acc"] = acc

                # PDF Report
                if st.button("📄 Generate PDF Report (Japan Gut)"):
                    if "japan_data" not in st.session_state:
                        st.error("⚠️ Please run predictions first!")
                    else:
                        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        pdf_file = tmp_pdf.name
                        tmp_pdf.close()

                        doc = SimpleDocTemplate(pdf_file)
                        styles = getSampleStyleSheet()
                        story = []

                        story.append(Paragraph("Japan Gut CAD Prediction Report", styles["Title"]))
                        story.append(Spacer(1, 20))
                        story.append(Paragraph(f"Accuracy: {st.session_state['japan_acc']:.2f}", styles["Normal"]))
                        story.append(Spacer(1, 10))

                        report = classification_report(
                            st.session_state["japan_data"]["Actual_Status"],
                            st.session_state["japan_data"]["Predicted_Status"],
                            output_dict=True
                        )
                        report_df = pd.DataFrame(report).transpose().round(2)
                        table_data = [["Class", "Precision", "Recall", "F1-Score", "Support"]]
                        for cls in report_df.index:
                            if cls == "accuracy": continue
                            row = [cls,
                                   report_df.loc[cls, "precision"],
                                   report_df.loc[cls, "recall"],
                                   report_df.loc[cls, "f1-score"],
                                   int(report_df.loc[cls, "support"])]
                            table_data.append(row)

                        tbl = Table(table_data, hAlign="LEFT")
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

                        for fig, title in zip([st.session_state["japan_fig1"], st.session_state["japan_fig2"]],
                                              ["Prediction Distribution", "Confusion Matrix"]):
                            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            fig.savefig(tmp_img.name, bbox_inches="tight", facecolor="white")
                            story.append(Paragraph(title, styles["Heading2"]))
                            story.append(Image(tmp_img.name, width=400, height=250))
                            story.append(Spacer(1, 20))
                            tmp_img.close()

                        doc.build(story)

                        with open(pdf_file, "rb") as f:
                            st.download_button(
                                label="📥 Download PDF",
                                data=f,
                                file_name="Japan_Gut_CAD_Report.pdf",
                                mime="application/pdf"
                            )

        except Exception as e:
            st.error(f"Error processing Japan Gut dataset: {e}")

# ---------------------------
# USA Custom Page
# ---------------------------
elif page == "USA Custom Page":
    st.markdown("Upload a CSV that includes a `CARDIOVASCULAR_DISEASE` column with 'Yes'/'No' values.")
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your gut microbiome CSV", type="csv", help="CSV must include CARDIOVASCULAR_DISEASE (Yes/No).")
    model_col, btn_col = st.columns([2, 1])
    with model_col:
        select_model = st.selectbox("Select Model", ["XGBoost", "SVM"], index=0)
    with btn_col:
        run_btn = st.button("Run CAD Prediction", key="run_pred", help="Run model predictions on the uploaded dataset")
        pdf_btn = st.button("📄 Generate PDF Report", key="gen_pdf")

    df = None
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as err:
            st.error(f"Could not read file: {err}")
            st.stop()
        
        if "CARDIOVASCULAR_DISEASE" not in df.columns:
            st.error("CSV must contain 'CARDIOVASCULAR_DISEASE' column with 'Yes'/'No' values.")
            st.stop()

        # keep responsibility of original code: drop VIOSCREEN_BMI if present
        if "VIOSCREEN_BMI" in df.columns:
            df = df.drop(columns=["VIOSCREEN_BMI"])

        # Show quick dataset KPIs
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown("<div class='kpi'><div class='value'>{}</div><div class='small-muted'>Rows</div></div>".format(df.shape[0]), unsafe_allow_html=True)
        with k2:
            st.markdown("<div class='kpi'><div class='value'>{}</div><div class='small-muted'>Features</div></div>".format(df.shape[1]), unsafe_allow_html=True)
        with k3:
            class_balance = df["CARDIOVASCULAR_DISEASE"].astype(str).str.lower().str.strip().value_counts().to_dict()
            balance_str = " / ".join([f"{k}:{v}" for k, v in class_balance.items()])
            st.markdown("<div class='kpi'><div class='value'>{}</div><div class='small-muted'>Class balance</div></div>".format(balance_str), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Preview & controls
        preview_tab, chart_tab, raw_tab = st.tabs(["Preview", "Charts", "Raw"])
        with preview_tab:
            st.subheader("Dataset preview")
            st.dataframe(df.head(100), use_container_width=True)

        with chart_tab:
            st.subheader("Feature & label overview")
            st.write("If you want richer charts, run the prediction and inspect results in the Results area.")

        with raw_tab:
            st.subheader("Raw CSV head")
            st.code(df.head(10).to_csv(index=False))

        # --- Load model & feature list (visual feedback) ---
        with st.spinner("Loading model and features..."):
            model = load_model(select_model.lower())
            feature_names = None
            try:
                feature_names = load_feature_names()
                st.success(f"Loaded {len(feature_names)} model features.")
            except Exception as e:
                st.warning(f"Could not load feature names: {e}")

        # Run prediction when user clicks the button
        if run_btn:
            preds = None
            try:
                preds = model.predict(df)
                st.success("Prediction successful (fully).")
            except Exception as e:
                st.warning(f"Pipeline preprocessing failed: {e}")
                st.info("Retrying by bypassing preprocessor...")

                if hasattr(model, "named_steps"):
                    try:
                        steps = list(model.named_steps.items())
                        final_step_name, final_estimator = steps[-1]
                        st.write(f"Detected final estimator: `{final_step_name}`")
                        df_numeric = df.select_dtypes(include=["number"]).copy()
                        preds = final_estimator.predict(df_numeric)
                        st.success("Prediction successful (bypassed pipeline).")
                    except Exception as e2:
                        st.error(f"Final estimator also failed: {e2}")
                else:
                    st.error("Model is not a Pipeline; cannot bypass preprocessing.")
            except ValueError as ve:
                st.error(f"Prediction failed: {ve}")

            # --- Show results ---
            if preds is not None:
                df_result = df.copy()
                df_result["prediction"] = preds

                label_map = {0: "cad", 1: "control"}
                pred_labels = [label_map.get(p, str(p).lower()) for p in preds]

                df["Predicted_Status"] = pred_labels
                y_real = df["CARDIOVASCULAR_DISEASE"].astype(str).str.lower().str.strip()
                df["Actual_Status"] = y_real

                # Save to session for PDF/export
                st.session_state.update({
                    "data_out": df,
                    "acc": None
                })

                # Basic charts & confusion matrix in tabs
                res_tab, cm_tab = st.tabs(["Prediction Breakdown", "Confusion Matrix"])
                with res_tab:
                    st.subheader("Predictions preview")
                    st.dataframe(df[["Predicted_Status", "Actual_Status"]].head(200), use_container_width=True)

                    pred_dist = pd.Series(pred_labels).value_counts()
                    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
                    sns.barplot(x=pred_dist.index, y=pred_dist.values, ax=ax1, palette="Set2")
                    ax1.set_title("Predicted Labels Distribution")
                    ax1.set_xlabel("Predicted Status")
                    ax1.set_ylabel("Count")
                    st.pyplot(fig1)

                with cm_tab:
                    def normalize_label(x):
                        x = str(x).strip().lower()
                        if x in ["yes", "1", "cad"]:
                            return "yes"
                        elif x in ["no", "0", "control"]:
                            return "no"
                        return x

                    y_real_norm = [normalize_label(v) for v in y_real]
                    pred_labels_norm = [normalize_label(v) for v in pred_labels]
                    unique_labels = sorted(set(y_real_norm) | set(pred_labels_norm))
                    cm = confusion_matrix(y_real_norm, pred_labels_norm, labels=unique_labels)

                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=unique_labels, yticklabels=unique_labels, ax=ax2)
                    ax2.set_xlabel("Predicted")
                    ax2.set_ylabel("Actual")
                    st.pyplot(fig2)

                    acc = accuracy_score(y_real_norm, pred_labels_norm)
                    st.markdown(f"### 🎯 Accuracy: **{acc:.2f}**")
                    # store things for PDF generation
                    st.session_state.update({
                        "fig1": fig1,
                        "fig2": fig2,
                        "acc": acc
                    })

    else:
        st.info("Upload a CSV to get started. You can use the sample dataset from the Datasets page.")

    # --- PDF Report generation (separate control) ---
    if pdf_btn:
        if "data_out" not in st.session_state:
            st.error("Please run predictions first.")
            st.stop()

        (Doc, Para, Spacer, RLImg, Table, TStyle, getStyles, colors) = bring_in_reportlab()

        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = Doc(temp_pdf.name)
        styles = getStyles()
        story = [Para("CAD Prediction Report", styles["Title"]), Spacer(1, 16)]

        if "acc" in st.session_state and st.session_state["acc"] is not None:
            story.append(Para(f"Model Accuracy: {st.session_state['acc']:.2f}", styles["Normal"]))
            story.append(Spacer(1, 10))

        report = classification_report(
            st.session_state["data_out"]["Actual_Status"],
            st.session_state["data_out"]["Predicted_Status"],
            output_dict=True
        )
        rep_df = pd.DataFrame(report).transpose().round(2)

        rows = [["Class", "Precision", "Recall", "F1", "Support"]]
        for label in rep_df.index:
            if label == "accuracy":
                continue
            row = [
                label,
                rep_df.loc[label, "precision"],
                rep_df.loc[label, "recall"],
                rep_df.loc[label, "f1-score"],
                int(rep_df.loc[label, "support"])
            ]
            rows.append(row)

        table = Table(rows, hAlign="LEFT")
        table.setStyle(TStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))

        story.append(Para("Classification Report", styles["Heading2"]))
        story.append(table)
        story.append(Spacer(1, 16))

        for fig_obj, fig_title in [("fig1", "Prediction Breakdown"), ("fig2", "Confusion Matrix")]:
            fig = st.session_state.get(fig_obj)
            if fig:
                tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmp_img.name, bbox_inches="tight", facecolor="white")
                story.append(Para(fig_title, styles["Heading2"]))
                story.append(RLImg(tmp_img.name, width=400, height=250))
                story.append(Spacer(1, 16))
                tmp_img.close()

        doc.build(story)

        with open(temp_pdf.name, "rb") as f:
            st.download_button("📥 Download PDF", data=f, file_name="CAD_Report.pdf", mime="application/pdf")


# ---------------------------
# Datasets Page
# ---------------------------
elif page == "Datasets":
    st.markdown("## 📂 Sample Datasets")
    sample_files = {
        "🍣 Japan Gut Microbiome": "PRJDB6472.csv",
        "🧬 USA Gut Microbiome": "dataset_filtered.csv",
    }
    for name, file in sample_files.items():
        if os.path.exists(file):
            with open(file, "rb") as f:
                st.download_button(label=f"⬇️ Download {name}", data=f, file_name=file, mime="text/csv")
        else:
            st.warning(f"{file} not found. Please add it to the app folder.")

# ---------------------------
# About Page
# ---------------------------
elif page == "About":
    st.markdown("## ℹ️ About this Project")
    st.write("""
    - **Japan Dataset** → Uses the existing Japan prediction code  
    - **USA Dataset** → Use the 'USA Custom Page' to paste your own processing code
    """)
