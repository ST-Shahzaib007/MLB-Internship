
# ============================ Mini Project =========================
#
# ==================== Unsupervised Model Training ==================
#
#
# ========================= Libraries Required ======================

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# ================================= IRIS data loading =============================
iris = load_iris()

df = pd.DataFrame(iris.data,
                  columns = iris.feature_names)
print("Head")
print(df.head())
print("Simple DF")
print(df)
print("Columns")        # +++++++++++  Dataset manuplutation
print(df.columns)
print("Info")
print(df.info())
print("Shape")
print(df.shape)

# ===================== KMean Algo with initial clusters 3 =======================
kmeans = KMeans(n_clusters = 3, random_state=42)

kmeans.fit(df)

clusters = kmeans.labels_

# =============== Within cluster Sum of Squares of simple KMeans =============
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df)
    wcss.append(round(kmeans.inertia_, 2))


# =================== Graphical Representation of WCSS of Simple IRIS dataset ===============
plt.plot(range(1, 11), wcss,marker='x')
plt.title("Elbow Methode")
plt.xlabel("Number of Clusters ")
plt.ylabel("WCSS ")
plt.show()

# ============= Cluster Visualization on the bases of Sepal length and width =============
plt.scatter(df["sepal length (cm)"],
            df["sepal width (cm)"],
            c=kmeans.labels_,
            cmap="cool")

plt.title("kmean clustring of iris data")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal width (cm)")

plt.show()

# =================== Cluster Visualization on the bases of Petal length and width ================
plt.figure(figsize= (8,6))
plt.scatter(df["petal length (cm)"],
            df["petal width (cm)"],
            c=kmeans.labels_,
            cmap="plasma")

plt.title("kmean clustring of iris data")
plt.xlabel("Petal Length (cm)")
plt.ylabel("petal width (cm)")

plt.show()

# ====================== Applying PCA for optimal 2 Features ==================
pca = PCA(n_components=2)

xpca = pca.fit_transform(df)

print(xpca.shape)
# ================ Converting Numpy array into pandas dataframe ===============
xpca_df = pd.DataFrame(
    xpca,
    columns=['PC1','PC2']
)
print(xpca_df.head())

# ============== Training of KMean on PCA features ==============
kmeans_pca = KMeans(n_clusters=3,random_state=42)
kmeans_pca.fit(xpca_df)


# ============== WCSS for Model Trained on PCA features =============
wcss_pca = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(xpca_df)
    wcss_pca.append(round(kmeans.inertia_,2))

print("**************** WCSS WITH 2 PCA Features and iris 4 features ******************* ")

print("\n"*2)
print("************ WCSS WITH original 4 iris Features *************** ")
print("WCSS     : ", wcss)    
print()

print("PCA WCSS : ",wcss_pca)
print("\n"*2)
# ==================== PCA Model trained Graph ================
plt.plot(range(1,11),wcss_pca,marker='x')
plt.title("PCA Graph ")
plt.xlabel("Number of clusters ")
plt.ylabel("WCSS PCA ")
plt.show()


# ======================== Graph to locate Data points within their clusters ================== 
plt.scatter(
    xpca_df['PC1'],
    xpca_df['PC2'],
    c = kmeans_pca.labels_, 
    cmap="viridis"
)

plt.title("Graph After PCA")
plt.xlabel("==== PC1 ====")
plt.ylabel("==== PC2 ====")

plt.show()

