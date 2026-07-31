import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="K-Means & PCA Demo",
    page_icon="🌸",
    layout="wide"
)

# ---------------------------------------------------
# Load Dataset (default Iris, or user-uploaded CSV)
# ---------------------------------------------------

st.sidebar.title("Navigation")

st.sidebar.subheader("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your own CSV (optional)",
    type=["csv"]
)

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    # Only keep numeric columns for clustering / PCA
    numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.sidebar.error(
            "Uploaded file needs at least 2 numeric columns. "
            "Falling back to the Iris dataset."
        )
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        target = iris.target
        using_iris = True
    else:
        df = df_raw[numeric_cols].dropna()
        target = None
        using_iris = False
        st.sidebar.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} numeric columns.")
else:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = iris.target
    using_iris = True

feature_names = df.columns.tolist()

option = st.sidebar.radio(
    "Choose Algorithm",
    (
        "K-Means Clustering",
        "PCA Visualization"
    )
)

# ===================================================
# K-MEANS
# ===================================================

if option == "K-Means Clustering":

    st.title("📊 K-Means Clustering")

    st.write(
        "This application groups your data into clusters using the K-Means algorithm."
    )

    st.subheader("Dataset")

    st.dataframe(df)

    k = st.slider(
        "Select Number of Clusters",
        min_value=2,
        max_value=10,
        value=3
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(df)

    result = df.copy()
    result["Cluster"] = clusters

    st.subheader("Clustered Dataset")

    st.dataframe(result)

    # -------------------------------
    # Feature selection for visualization
    # -------------------------------

    st.subheader("Cluster Visualization")

    col_x, col_y = st.columns(2)

    with col_x:
        x_feature = st.selectbox(
            "X-axis feature",
            feature_names,
            index=min(2, len(feature_names) - 1)
        )

    with col_y:
        y_feature = st.selectbox(
            "Y-axis feature",
            feature_names,
            index=min(3, len(feature_names) - 1)
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        df[x_feature],
        df[y_feature],
        c=clusters,
        cmap="viridis",
        s=70
    )

    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title("K-Means Clustering")

    st.pyplot(fig)

    # -------------------------------
    # User Input for Prediction
    # -------------------------------

    st.subheader("Predict Cluster for a New Sample")

    with st.form("cluster_form"):

        st.write("Enter values for each feature:")

        input_cols = st.columns(2)
        user_values = {}

        for i, feat in enumerate(feature_names):
            default_val = float(df[feat].mean())
            with input_cols[i % 2]:
                user_values[feat] = st.number_input(
                    feat,
                    value=round(default_val, 2),
                    key=f"km_{feat}"
                )

        predict = st.form_submit_button("Predict Cluster")

    if predict:

        sample = [[user_values[feat] for feat in feature_names]]

        cluster = kmeans.predict(sample)

        st.success(f"Sample belongs to Cluster {cluster[0]}")

        if using_iris:
            st.info(
                "K-Means is an Unsupervised Learning algorithm. "
                "It does not predict the flower species. "
                "It only assigns the flower to the nearest cluster."
            )

        fig2, ax2 = plt.subplots(figsize=(8, 6))

        ax2.scatter(
            df[x_feature],
            df[y_feature],
            c=clusters,
            cmap="viridis",
            s=70,
            label="Dataset"
        )

        ax2.scatter(
            user_values[x_feature],
            user_values[y_feature],
            color="red",
            marker="X",
            s=220,
            label="Your Sample"
        )

        ax2.set_xlabel(x_feature)
        ax2.set_ylabel(y_feature)
        ax2.set_title("Cluster Prediction")
        ax2.legend()

        st.pyplot(fig2)

    # ---------------------------------
    # Elbow Method
    # ---------------------------------

    st.subheader("Elbow Method")

    show_elbow = st.checkbox("Show Elbow Method plot", value=True)

    if show_elbow:

        max_k = st.slider(
            "Max clusters to test for elbow method",
            min_value=3,
            max_value=15,
            value=10
        )

        wcss = []

        for i in range(1, max_k + 1):

            model = KMeans(
                n_clusters=i,
                random_state=42,
                n_init=10
            )

            model.fit(df)

            wcss.append(model.inertia_)

        fig3, ax3 = plt.subplots(figsize=(8, 5))

        ax3.plot(
            range(1, max_k + 1),
            wcss,
            marker="o"
        )

        ax3.set_xlabel("Number of Clusters")
        ax3.set_ylabel("WCSS")
        ax3.set_title("Elbow Method")

        st.pyplot(fig3)

# ===================================================
# PCA
# ===================================================

else:

    st.title("📉 Principal Component Analysis (PCA)")

    st.write(
        "Principal Component Analysis (PCA) reduces the dimensionality "
        "of the dataset while preserving most of its information."
    )

    st.subheader("Original Dataset")

    st.dataframe(df)

    # ---------------------------------
    # Apply PCA
    # ---------------------------------

    max_components = min(len(feature_names), df.shape[0])

    n_components = st.slider(
        "Number of Principal Components",
        min_value=2,
        max_value=max(2, max_components),
        value=2
    )

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)

    pca_df = pd.DataFrame(
        components,
        columns=[f"Principal Component {i+1}" for i in range(n_components)]
    )

    st.subheader("PCA Transformed Dataset")

    st.dataframe(pca_df)

    # ---------------------------------
    # PCA Visualization
    # ---------------------------------

    st.subheader("PCA Visualization")

    pc_cols = st.columns(2)

    with pc_cols[0]:
        pc_x = st.selectbox(
            "X-axis component",
            pca_df.columns.tolist(),
            index=0
        )

    with pc_cols[1]:
        pc_y = st.selectbox(
            "Y-axis component",
            pca_df.columns.tolist(),
            index=min(1, len(pca_df.columns) - 1)
        )

    fig4, ax4 = plt.subplots(figsize=(8, 6))

    if using_iris:
        scatter = ax4.scatter(
            pca_df[pc_x],
            pca_df[pc_y],
            c=target,
            cmap="viridis",
            s=70
        )
    else:
        scatter = ax4.scatter(
            pca_df[pc_x],
            pca_df[pc_y],
            s=70
        )

    ax4.set_xlabel(pc_x)
    ax4.set_ylabel(pc_y)
    ax4.set_title("PCA Visualization")

    st.pyplot(fig4)

    # ---------------------------------
    # User Input for Transformation
    # ---------------------------------

    st.subheader("Transform a New Sample")

    with st.form("pca_form"):

        st.write("Enter values for each feature:")

        input_cols = st.columns(2)
        user_values = {}

        for i, feat in enumerate(feature_names):
            default_val = float(df[feat].mean())
            with input_cols[i % 2]:
                user_values[feat] = st.number_input(
                    feat,
                    value=round(default_val, 2),
                    key=f"pca_{feat}"
                )

        transform = st.form_submit_button("Transform Using PCA")

    if transform:

        sample = [[user_values[feat] for feat in feature_names]]

        transformed = pca.transform(sample)

        st.success("Transformation Successful!")

        for i in range(n_components):
            st.write(f"**Principal Component {i+1}:** {transformed[0][i]:.3f}")

        fig5, ax5 = plt.subplots(figsize=(8, 6))

        if using_iris:
            ax5.scatter(
                pca_df[pc_x],
                pca_df[pc_y],
                c=target,
                cmap="viridis",
                s=70,
                label="Dataset"
            )
        else:
            ax5.scatter(
                pca_df[pc_x],
                pca_df[pc_y],
                s=70,
                label="Dataset"
            )

        x_idx = int(pc_x.split(" ")[-1]) - 1
        y_idx = int(pc_y.split(" ")[-1]) - 1

        ax5.scatter(
            transformed[0][x_idx],
            transformed[0][y_idx],
            color="red",
            marker="X",
            s=250,
            label="Your Sample"
        )

        ax5.set_xlabel(pc_x)
        ax5.set_ylabel(pc_y)
        ax5.set_title("PCA Visualization with Your Input")
        ax5.legend()

        st.pyplot(fig5)

    # ---------------------------------
    # Explained Variance
    # ---------------------------------

    st.subheader("Explained Variance Ratio")

    variance = pd.DataFrame(
        {
            "Principal Component": [f"PC{i+1}" for i in range(n_components)],
            "Explained Variance Ratio": pca.explained_variance_ratio_
        }
    )

    st.dataframe(variance)

    st.info(
        f"Together, these {n_components} component(s) explain "
        f"{pca.explained_variance_ratio_.sum()*100:.1f}% of the total variance in the data."
    )
