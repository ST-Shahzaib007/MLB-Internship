import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

st.set_page_config(page_title="Breast Cancer Prediction System")
with st.sidebar:
    st.title("🩺 Breast Cancer ML")
    st.write("Day 10 Internship Project")
    st.markdown("---")
    st.success("Upload Dataset")
    st.info("Train Models")
    st.warning("Compare Results")

st.markdown(
    "<h1 style='text-align:center;color:#2E8B57;'>🩺 Breast Cancer Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Upload the Breast Cancer Wisconsin dataset in CSV format.</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)



if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    with st.expander("📄 Dataset Preview"):
        st.dataframe(df.head())



    with st.expander("ℹ Dataset Information"):

        st.write("Shape:", df.shape)

        st.write("Columns:")

        st.write(list(df.columns))

    with st.expander("📊 Statistical Summary"):

        st.write(df.describe())


    # Features and Target
    X = df.drop("target", axis=1)
    y = df["target"]  


    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
    )


    if st.button("Train Models"):
        baseline_model = LogisticRegression(max_iter=10000)

        baseline_model.fit(X_train, y_train)

        baseline_pred = baseline_model.predict(X_test)    

        st.subheader("Baseline Model")

        st.write("Accuracy:", accuracy_score(y_test, baseline_pred))
        st.write("Precision:", precision_score(y_test, baseline_pred))
        st.write("Recall:", recall_score(y_test, baseline_pred))
        st.write("F1 Score:", f1_score(y_test, baseline_pred))


        param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"]
        }

        grid = GridSearchCV(
            LogisticRegression(max_iter=10000),
            param_grid,
            cv=5,
            scoring="accuracy"
        )

        with st.spinner("Training and tuning model..."):
            grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        st.success("Models trained successfully!")

        tuned_pred = best_model.predict(X_test)
        # Best Hyperparameters
        st.subheader("Best Hyperparameters")
        st.write(grid.best_params_)

        # Tuned Model
        st.subheader("Tuned Model")
        st.subheader("Baseline Model")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy_score(y_test, baseline_pred):.4f}")
        col2.metric("Precision", f"{precision_score(y_test, baseline_pred):.4f}")
        col3.metric("Recall", f"{recall_score(y_test, baseline_pred):.4f}")
        col4.metric("F1 Score", f"{f1_score(y_test, baseline_pred):.4f}")

        st.subheader("Baseline Classification Report")

        st.text(classification_report(y_test, baseline_pred))

        st.subheader("Tuned Classification Report")

        st.text(classification_report(y_test, tuned_pred))

        st.subheader("Model Comparison")

        comparison = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Baseline": [
                accuracy_score(y_test, baseline_pred),
                precision_score(y_test, baseline_pred),
                recall_score(y_test, baseline_pred),
                f1_score(y_test, baseline_pred)
            ],
            "Tuned": [
                accuracy_score(y_test, tuned_pred),
                precision_score(y_test, tuned_pred),
                recall_score(y_test, tuned_pred),
                f1_score(y_test, tuned_pred)
            ]
        })

        st.dataframe(comparison)

        baseline_cm = confusion_matrix(y_test, baseline_pred)
        tuned_cm = confusion_matrix(y_test, tuned_pred)


        st.subheader("Confusion Matrices")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(
            baseline_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0]
        )

        axes[0].set_title("Baseline Model")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        sns.heatmap(
            tuned_cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            ax=axes[1]
        )

        axes[1].set_title("Tuned Model")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")

        st.pyplot(fig)
        st.balloons()