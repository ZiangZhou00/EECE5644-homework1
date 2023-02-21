# Reference from Mark Zolotas
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
np.set_printoptions(suppress=True)
# Set seed
np.random.seed(7)
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 18,
    'figure.titlesize': 22
})
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
     [0, 0, 0, 3]]
])
n = mu.shape[1]
# Class priors
priors = np.array([0.65, 0.35])
C = len(priors)
# Caculate threshold rule
Lambda = np.ones((C, C)) - np.identity(C)
gamma = (Lambda[1,0] - Lambda[0,0])/(Lambda[0,1] - Lambda[1,1]) * priors[0] / priors[1]
print(f'Threshold value: {gamma}')
u = np.random.rand(N)
#threshold = np.linspace(0,10,100)
# Output samples and labels
X = np.zeros([N, n])
labels = np.zeros(N) # KEEP TRACK OF THIS
# Plot for original data and their true labels
labels = np.random.rand(N) >= priors[0]
L = np.array(range(C))
Nl = np.array([sum(labels == l) for l in L])
print("Number of samples from Class 1: {:d}, Class 2: {:d}".format(Nl[0], Nl[1]))
X = np.zeros((N, n))
X[labels == 0, :] =  multivariate_normal.rvs(mu[0], Sigma[0], Nl[0])
X[labels == 1, :] =  multivariate_normal.rvs(mu[1], Sigma[1], Nl[1])
class_conditional_likelihoods = np.array([multivariate_normal.pdf(X, mu[l], Sigma[l]) for l in L])
discriminant_score_erm = np.log(class_conditional_likelihoods[1]) - np.log(class_conditional_likelihoods[0])
print("class conditional: ",discriminant_score_erm)
gamma_map = (Lambda[1,0] - Lambda[0,0]) / (Lambda[0,1] - Lambda[1,1]) * priors[0]/priors[1]
from sys import float_info # Threshold smallest positive floating value
# Generate ROC curve samples
def estimate_roc(discriminant_score, label):
    Nlabels = np.array((sum(label == 0), sum(label == 1)))
    sorted_score = sorted(discriminant_score)
    # Use tau values that will account for every possible classification split
    taus = ([sorted_score[0] - float_info.epsilon] +
            sorted_score +
            [sorted_score[-1] + float_info.epsilon])
    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= t for t in taus]
    ind10 = [np.argwhere((d==1) & (label==0)) for d in decisions]
    p10 = [len(inds)/Nlabels[0] for inds in ind10]
    ind11 = [np.argwhere((d==1) & (label==1)) for d in decisions]
    p11 = [len(inds)/Nlabels[1] for inds in ind11]

    ind01 = [np.argwhere((d==0) & (label==1)) for d in decisions]
    p01 = [len(inds)/Nlabels[1] for inds in ind01]
    prob_error_erm = np.zeros(len(p01))
    for i in range(len(p10)):
        prob_error_erm[i] = np.array((p10[i], p01[i])).dot(Nlabels.T / N)
    best_gamma = np.exp(taus[np.argmin(prob_error_erm)])
    p_error_erm = min(prob_error_erm)
    # ROC has FPR on the x-axis and TPR on the y-axis
    roc = np.array((p10, p11))
    return roc, taus, best_gamma, p_error_erm
gamma_map = priors[0]/priors[1]
decisions_map = discriminant_score_erm >= np.log(gamma_map)
ind_00_map = np.argwhere((decisions_map==0) & (labels==0))
p_00_map = len(ind_00_map) / Nl[0]
ind_10_map = np.argwhere((decisions_map==1) & (labels==0))
p_10_map = len(ind_10_map) / Nl[0]
ind_01_map = np.argwhere((decisions_map==0) & (labels==1))
p_01_map = len(ind_01_map) / Nl[1]
ind_11_map = np.argwhere((decisions_map==1) & (labels==1))
p_11_map = len(ind_11_map) / Nl[1]
prob_error_erm = np.array((p_10_map, p_01_map)).dot(Nl.T / N)
# Construct the ROC for ERM by changing log(gamma)
roc_erm, _, min_prob_error, taus = estimate_roc(discriminant_score_erm, labels)
roc_map = np.array((p_10_map, p_11_map))
prob_error = np.array((roc_erm[0, :], 1 - roc_erm[1, :])).T.dot(Nl.T / np.sum(Nl))
min_prob_error = np.min(prob_error)
print('Min Empirical Pr(error) for ERM: ', min_prob_error)
print('Min Theoretical Pr(error) for ERM: ', prob_error_erm)
print('Min Theoretical Gamma: ', gamma_map)
print('Min Empirical Gamma: ', taus)
fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
ax_roc.plot(roc_erm[0], roc_erm[1])
ax_roc.plot(roc_map[0], roc_map[1], 'rx', label="Naive Min Pr(error)ERM", markersize=16)
ax_roc.legend()
ax_roc.set_xlabel(r"Probability of false alarm $P(D=1|L=0)$")
ax_roc.set_ylabel(r"Probability of correct decision $P(D=1|L=1)$")
plt.title('Naive Empirical ROC')
plt.grid(True)
plt.show()
