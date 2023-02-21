# Reference from Mark Zolotas
import math
from Functions import *
np.set_printoptions(suppress=True)
# Set seed
N = 10000
mu = np.array([[-1/2, -1/2, -1/2, -1/2], [1, 1, 1, 1]])
Sigma = np.array([
    [[0.5, -0.125, 0.075, 0],
     [-0.125, 0.25, -0.125, 0],
     [0.075, -0.125, 0.25, 0],
     [0, 0, 0, 0.5]],
    [[1, 0.3, -0.2, 0],
     [0.3, 1, 0.3, 0],
     [-0.2, 0.3, 1, 0],
     [0, 0, 0, 3]]])
n = mu.shape[1]
priors = np.array([0.65, 0.35])
C = len(priors)
labels = np.random.rand(N) >= priors[0]
L = np.arange(C)
Nl = np.array([np.sum(labels == l) for l in L])
X = np.zeros((N, n))
X[labels == 0, :] = multivariate_normal.rvs(mu[0], Sigma[0], Nl[0])
X[labels == 1, :] = multivariate_normal.rvs(mu[1], Sigma[1], Nl[1])
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(X[labels == 0, 0], X[labels == 0, 1], 'bo', label="Class 0")
ax.plot(X[labels == 1, 0], X[labels == 1, 1], 'k+', label="Class 1")
ax.legend()
ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$")
plt.tight_layout()
plt.show()
Lambda = np.ones((C, C)) - np.identity(C)
class_conditional_likelihoods = np.array([multivariate_normal.pdf(X, mu[l], Sigma[l]) for l in L])
discriminant_score_erm = np.log(class_conditional_likelihoods[1]) - np.log(class_conditional_likelihoods[0])
gamma_map = priors[0] / priors[1]
decisions_map = discriminant_score_erm >= np.log(gamma_map)
Nl = np.array([np.sum(labels == 0), np.sum(labels == 1)])
ind_00_map = np.argwhere((decisions_map == 0) & (labels == 0))
ind_10_map = np.argwhere((decisions_map == 1) & (labels == 0))
ind_01_map = np.argwhere((decisions_map == 0) & (labels == 1))
ind_11_map = np.argwhere((decisions_map == 1) & (labels == 1))
p_00_map = len(ind_00_map) / Nl[0]
p_10_map = len(ind_10_map) / Nl[0]
p_01_map = len(ind_01_map) / Nl[1]
p_11_map = len(ind_11_map) / Nl[1]
prob_error_erm = np.array((p_10_map, p_01_map)).dot(Nl.T / N)
fig = plt.figure(figsize=(10, 10))
plt.plot(X[ind_00_map, 0], X[ind_00_map, 1], 'og', label="Correct Class 0")
plt.plot(X[ind_10_map, 0], X[ind_10_map, 1], 'or', label="Incorrect Class 0")
plt.plot(X[ind_01_map, 0], X[ind_01_map, 1], '+r', label="Incorrect Class 1")
plt.plot(X[ind_11_map, 0], X[ind_11_map, 1], '+g', label="Correct Class 1")
plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("MAP Decisions")
plt.tight_layout()
plt.show()
conf_mat = confusion_matrix(decisions_map, labels)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Confusion Matrix (rows: Predicted class, columns: True class):\n", conf_mat)
print("Total Number of Misclassified Samples: {:d}".format(N - correct_class_samples))
from sys import float_info
epsilon = float_info.epsilon
# Construct the ROC for ERM by changing log(gamma)
roc_erm, taus = estimate_roc(discriminant_score_erm, labels)
roc_map = np.array((p_10_map, p_11_map))
Nl = np.array([sum(labels == 0), sum(labels == 1)])
prob_error = np.array((roc_erm[0, :], 1 - roc_erm[1, :])).T.dot(Nl.T / np.sum(Nl))
min_prob_error = np.min(prob_error)
min_ind = np.argmin(prob_error)
print("Min Empirical Pr(error) for ERM: {:.4f}".format(min_prob_error))
print("Min Theoretical Pr(error) for ERM: {:.4f}".format(prob_error_erm))
print("Min Theoretical Gamma: {:.4f}".format(gamma_map))
print("Min Empirical Gamma: {:.4f}".format(math.exp(taus[min_ind])))
fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
ax_roc.plot(roc_erm[0], roc_erm[1])
ax_roc.plot(roc_map[0], roc_map[1], 'rx', label="Min Theoretical Pr(error) ERM", markersize=16)
ax_roc.plot(roc_erm[0, min_ind], roc_erm[1, min_ind], 'ro', label="Min Empirical Pr(error) ERM", markersize=16)
ax_roc.legend()
ax_roc.set_xlabel(r"Probability of false alarm $P(D=1|L=0)$")
ax_roc.set_ylabel(r"Probability of correct decision $P(D=1|L=1)$")
plt.grid(True)
plt.show()
