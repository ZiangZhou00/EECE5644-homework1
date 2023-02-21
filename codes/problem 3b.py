# Reference from Mark Zolotas
from Functions import *


from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

# Reads zip file without writing to disk by emulating the file using a BytesIO buffer class
resp = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip')
har_zip = ZipFile(BytesIO(resp.read()))
har_train_df = pd.read_csv(har_zip.open('UCI HAR Dataset/train/X_train.txt'), delim_whitespace=True, header=None)
har_test_df = pd.read_csv(har_zip.open('UCI HAR Dataset/test/X_test.txt'), delim_whitespace=True, header=None)
har_df = pd.concat([har_train_df, har_test_df])

# Extracting data matrix X
X = har_df.to_numpy()

fig, ax_pca = plt.subplots(figsize=(10, 10))

pca = PCA(n_components=2)  # project data onto first two principal components
X_fit = pca.fit(X)  # fit PCA object to the data
Z = pca.transform(X)  # transform data into 2D space

# print the fraction of the total variance in the data that is captured by each of the two principal components
print("Explained variance ratio: ", pca.explained_variance_ratio_)

# create a scatter plot of the data in the 2D space defined by the two principal components
ax_pca.scatter(Z[:, 0], Z[:, 1])

# add labels to the axes of the 2D scatter plot
ax_pca.set_xlabel(r"$z_1$")
ax_pca.set_ylabel(r"$z_2$")


plt.title("PCA of Human Activity Recognition Dataset")
plt.tight_layout()
plt.show()