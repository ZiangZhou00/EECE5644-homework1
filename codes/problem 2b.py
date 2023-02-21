# Reference from Mark Zolotas
from Functions import *

N = 10000
n = 3 # dimensionality of input random vectors
C = 3 # number of classes

gmm_pdf = {}

# Class priors
gmm_pdf['priors'] = np.array([0.3, 0.3, 0.4])

# Set mean vectors to be equally spaced out along a line in order
gmm_pdf['mu'] = np.array([1*np.ones(n), 2*np.ones(n), 3*np.ones(n)])

# Set covariance matrices so that there is significant overlap between these
# distributions with means given above
gmm_pdf['Sigma'] = np.array([2*np.eye(n), 2*np.eye(n), 2*np.eye(n)])
X, labels = generate_data_from_gmm(N, gmm_pdf)
L = np.array(range(C)) # Assuming 0-2 instead of 1-3 to make my life easier in Python

# Create the new loss matrix
Lambda = np.array([[0, 1, 10],
                   [1, 0, 10],
                   [1, 1, 0]])

# ERM decision rule, take index/label associated with minimum conditional risk as decision (N, 1)
decisions = perform_erm_classification(X, Lambda, gmm_pdf, C)

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, labels)
conf_display = ConfusionMatrixDisplay.from_predictions(decisions, labels, display_labels=['1', '2', '3'],
                                                       colorbar=False)
plt.ylabel("Predicted Labels")
plt.xlabel("True Labels")

correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples))

prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))


# Plot for decisions vs true labels
fig_map = plt.figure(figsize=(10, 10))
ax = fig_map.add_subplot(111, projection='3d')

marker_shapes = '.o^s'
for r in L: # Each decision option
    for c in L: # Each class label
        ind_rc = np.argwhere((decisions==r) & (labels==c))

        # Decision = Marker Shape; True Labels = Marker Color
        if r == c:
            marker = marker_shapes[r] + 'g'
            plt.plot(X[ind_rc, 0], X[ind_rc, 1], marker,
                     label="D = {} | L = {}".format(r+1, c+1), markerfacecolor='none')
        else:
            marker = marker_shapes[r] + 'r'
            plt.plot(X[ind_rc, 0], X[ind_rc, 1], marker,
                     label="D = {} | L = {}".format(r+1, c+1), markerfacecolor='none')

ax.legend()
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
plt.title("Classification Decisions: Marker Shape/Predictions, Color/True Labels")
plt.tight_layout()
plt.show()

