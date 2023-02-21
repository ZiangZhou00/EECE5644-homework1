# Reference from Mark Zolotas
from Functions import *


wine_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
                       delimiter=';')

def regularized_cov(X, lambda_reg):
    n = X.shape[0]
    sigma = np.cov(X)
    # Selecting the regularization parameter should be performed using cross-validation and a separate data subset
    # As I only went by training set performance (overfitting) in this problem, I settled on lambda=1/n
    sigma += lambda_reg * np.eye(n)
    return sigma

# Total number of rows/samples
N = len(wine_df.index)

# iloc accesses rows/columns by indexing
# Extracting data matrix X and target labels vector
X = wine_df.iloc[:, :-1].to_numpy()
qualities = wine_df.iloc[:, -1].to_numpy()

# Define a label encoder to make indexing easy and encode labels as 0, 1, ..., C, rather than 4, 5, ... etc
le = preprocessing.LabelEncoder()
le.fit(qualities)
labels = le.transform(qualities)

# Estimate class priors
gmm = {'priors': (wine_df.groupby(['quality']).size() / N).to_numpy()}
# Infer number of classes from priors
num_classes = len(gmm['priors'])

gmm['mu'] = wine_df.groupby(['quality']).mean().to_numpy()
# Infer number of features from priors
n = gmm['mu'].shape[1]

gmm['Sigma'] = np.array([regularized_cov(X[labels == l].T, (1/n)) for l in range(num_classes)])

N_per_l = np.array([sum(labels == l) for l in range(num_classes)])
print(N_per_l)


# Total number of rows/samples
N = len(wine_df.index)

# iloc accesses rows/columns by indexing
# Extracting data matrix X and target labels vector
X = wine_df.iloc[:, :-1].to_numpy()
qualities = wine_df.iloc[:, -1].to_numpy()

# Define a label encoder to make indexing easy and encode labels as 0, 1, ..., C, rather than 4, 5, ... etc
le = preprocessing.LabelEncoder()
le.fit(qualities)
labels = le.transform(qualities)


# If 0-1 loss then yield MAP decision rule, else ERM classifier
Lambda = np.ones((num_classes, num_classes)) - np.eye(num_classes)

# ERM decision rule, take index/label associated with minimum conditional risk as decision (N, 1)
decisions = perform_erm_classification(X, Lambda, gmm, num_classes)

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
plt.show()
conf_mat = confusion_matrix(decisions, labels)
fig, ax = plt.subplots(figsize=(10, 10))
conf_display = ConfusionMatrixDisplay.from_predictions(decisions, labels, ax=ax,
                                                       display_labels=['3', '4', '5', '6', '7', '8', '9'], colorbar=True)
plt.ylabel('Predicted Labels')
plt.xlabel('True Labels')

correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Number of Misclassified Samples: {:d}".format(N - correct_class_samples))

prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))
plt.show()

fig = plt.figure(figsize=(10, 10))

ax_subset = fig.add_subplot(111, projection='3d')

unique_qualities = np.sort(wine_df['quality'].unique())
for q in range(unique_qualities[0], unique_qualities[-1]):
    ax_subset.scatter(wine_df[wine_df['quality'] == q]['residual sugar'],
                      wine_df[wine_df['quality'] == q]['chlorides'],
                      wine_df[wine_df['quality'] == q]['density'], label="Quality {}".format(q))

ax_subset.set_xlabel("residual sugar")
ax_subset.set_ylabel("chlorides")
ax_subset.set_zlabel("density")

# Set equal axes for 3D plots to realize the additional challenges in visualization
# ax_subset.set_box_aspect((np.ptp(wine_df['fixed acidity']), np.ptp(wine_df['alcohol']), np.ptp(wine_df['pH'])))

plt.title("Wine Subset of Features(residual sugar, chlorides and density)")
plt.legend()
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(10, 10))

ax_pca = fig.add_subplot(111, projection='3d')

pca = PCA(n_components=3)  # n_components is how many PCs we'll keep
X_fit = pca.fit(X)  # Is a fitted estimator, not actual data to project
Z = pca.transform(X)

# Illustrates that we have at least 90% of the total variance captured by the PCA
print("Explained variance ratio: ", pca.explained_variance_ratio_)

for q in range(unique_qualities[0], unique_qualities[-1]):
    ax_pca.scatter(Z[wine_df['quality'] == q, 0],
                   Z[wine_df['quality'] == q, 1],
                   Z[wine_df['quality'] == q, 2], label="Quality {}".format(q))

ax_pca.set_xlabel(r"$z_1$")
ax_pca.set_ylabel(r"$z_2$")
ax_pca.set_zlabel(r"$z_3$")

ax_pca.set_box_aspect((np.ptp(Z[:, 0]), np.ptp(Z[:, 1]), np.ptp(Z[:, 2])))

plt.title("PCA of Wine Dataset")
plt.legend()
plt.tight_layout()
plt.show()
