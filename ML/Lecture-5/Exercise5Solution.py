# %%
# ###################################
# Group ID : 741
# Members : Nikolaj Ask Albertsen, Rui Maria Martins Loureiro, Bjarki Fróðason Í Eyðansstovu,
#  Magnus Vørs Holmsgaard, Gustav Søndergaard Nybro
# Date : 07/10 - 2026
# Lecture: 5 - Clustering
# Dependencies: scipy, numpy, matplotlib, sklearn
# Python version: 3.13
# Functionality: Fits a Gaussian Mixture Model to combined, unlabelled 2D MNIST data
# for classes 5, 6 and 8. 
# ###################################

from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal as norm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

# %% [markdown]
# # Exercise 5: Clustering
# This assignment is based on the previously generated 2-dimensional data
# of the three classes (5, 6 and 8) from the MNIST database of handwritten digits.
#
# First, mix the training data by removing the labels and use one
# Gaussian mixture model to model them.
#
# Secondly, compare the GMM with Gaussian models trained separately
# for each class, using means, covariances and visualisation.

# %% [markdown]
# ## Loading the data and mixing
# Load the dataset, combine the training sets and shuffle the data.
# Use a seed for reproducibility.

# %%
data_path = "2D568class.mat"
data = loadmat(data_path)

train5 = data["trn5_2dim"] / 255
train6 = data["trn6_2dim"] / 255
train8 = data["trn8_2dim"] / 255

trainset = np.concatenate([train5, train6, train8], axis=0)

np.random.seed(0)
np.random.shuffle(trainset)

# %% [markdown]
# ## Creating a Gaussian Mixture model
# Create and fit a GMM with three components using sklearn.

# %%
gmm = GMM(
    n_components=3,
    covariance_type="full",
    n_init=10,
    random_state=42
)

gmm.fit(trainset)

# %% [markdown]
# ## Creating Gaussian models
# Estimate a mean and covariance for each known class.
# Use these parameters to create one Gaussian model per class.

# %%
mean5 = np.mean(train5, axis=0)
mean6 = np.mean(train6, axis=0)
mean8 = np.mean(train8, axis=0)

cov5 = np.cov(train5, rowvar=False, bias=True)
cov6 = np.cov(train6, rowvar=False, bias=True)
cov8 = np.cov(train8, rowvar=False, bias=True)

gaussian5 = norm(mean=mean5, cov=cov5)
gaussian6 = norm(mean=mean6, cov=cov6)
gaussian8 = norm(mean=mean8, cov=cov8)

class_means = np.array([mean5, mean6, mean8])
class_covs = np.array([cov5, cov6, cov8])
class_models = [gaussian5, gaussian6, gaussian8]

# %% [markdown]
# ## Comparing means and covariance matrices
# First, extract the means and covariances from the GMM.

# %%
gmm_means = gmm.means_
gmm_covs = gmm.covariances_

# %% [markdown]
# Match the GMM components to classes 5, 6 and 8 for comparison.
# Choose the one-to-one assignment with the smallest total distance
# between class means and GMM means.
#
# This matching is only for comparison; it does not retrain the GMM
# or guarantee that the components correspond to the digit classes.

# %%
distances = np.linalg.norm(
    class_means[:, None, :] - gmm_means[None, :, :],
    axis=2
)

class_indices, component_indices = linear_sum_assignment(distances)
order = component_indices[np.argsort(class_indices)]

gmm_means = gmm_means[order]
gmm_covs = gmm_covs[order]

mean1_gmm, mean2_gmm, mean3_gmm = gmm_means
cov1_gmm, cov2_gmm, cov3_gmm = gmm_covs

# %% [markdown]
# ### Means
# Print each class mean alongside its matched GMM component mean.

# %%
digits = [5, 6, 8]

for i, digit in enumerate(digits):
    print(f"\nClass {digit} vs GMM component {order[i]}")
    print("Class mean:", np.array2string(class_means[i], precision=5))
    print("GMM mean:  ", np.array2string(gmm_means[i], precision=5))

# %% [markdown]
# ### Covariances
# Display class covariance matrices in the top row and the matched
# GMM covariance matrices in the bottom row.
# Use a shared colour scale to make the matrices comparable.

# %%
fig, axs = plt.subplots(
    2, 3, figsize=(15, 8), constrained_layout=True
)

vmin = min(class_covs.min(), gmm_covs.min())
vmax = max(class_covs.max(), gmm_covs.max())

for column, digit in enumerate(digits):
    matrices = [class_covs[column], gmm_covs[column]]
    titles = [
        f"Covariance: class {digit}",
        f"Covariance: GMM component {order[column]}"
    ]

    for row, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = axs[row, column]

        im = ax.matshow(
            matrix,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax
        )

        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j, i, f"{value:.4g}",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )

        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8)
plt.show()

# %% [markdown]
# What do we see when comparing means and covariances?
#
# Compare the class means with the matched GMM means to assess how
# closely their centres agree.
# Compare the covariance diagonals for feature variances and the
# off-diagonal entries for how the two features vary together.
# Write your conclusions after inspecting the results.

# %% [markdown]
# ## Visualizing the models in contourplots
# First, generate a grid of points at which to evaluate the densities.

# %%
padding = 0.1 * np.ptp(trainset, axis=0)
lower = trainset.min(axis=0) - padding
upper = trainset.max(axis=0) + padding

xx, yy = np.meshgrid(
    np.linspace(lower[0], upper[0], 200),
    np.linspace(lower[1], upper[1], 200)
)

points = np.column_stack([xx.ravel(), yy.ravel()])

# %% [markdown]
# Create separate Gaussian models from the matched GMM component
# means and covariances.

# %%
component_models = [
    norm(mean=mean, cov=cov)
    for mean, cov in zip(gmm_means, gmm_covs)
]

# %% [markdown]
# Evaluate the full GMM, the separate GMM components and the classwise
# Gaussian models at the generated points.
# These are density evaluations, not randomly generated samples.

# %%
# score_samples returns log density.
mixture_density = np.exp(
    gmm.score_samples(points)
).reshape(xx.shape)

component_densities = [
    model.pdf(points).reshape(xx.shape)
    for model in component_models
]

class_densities = [
    model.pdf(points).reshape(xx.shape)
    for model in class_models
]

# %% [markdown]
# Visualize the evaluated densities in contour plots.
# Plot the full GMM, separate GMM components and classwise Gaussians.
# Use matching colours and contour levels for each matched pair.
# The separate component plots show unweighted Gaussian densities;
# the full mixture includes the learned mixture weights.

# %%
fig, axs = plt.subplots(
    1, 3, figsize=(17, 5), sharex=True, sharey=True
)

colors = ["tab:blue", "tab:orange", "tab:green"]
training_sets = [train5, train6, train8]

axs[0].scatter(
    trainset[:, 0], trainset[:, 1],
    s=5, alpha=0.15, color="gray"
)
axs[0].contour(xx, yy, mixture_density, levels=8)
axs[0].set_title("Full Gaussian mixture")

for i, (digit, samples, color) in enumerate(
    zip(digits, training_sets, colors)
):
    peak = max(
        component_densities[i].max(),
        class_densities[i].max()
    )
    levels = peak * np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    axs[1].contour(
        xx, yy, component_densities[i],
        levels=levels, colors=[color]
    )

    axs[2].scatter(
        samples[:, 0], samples[:, 1],
        s=5, alpha=0.2, color=color, label=str(digit)
    )
    axs[2].contour(
        xx, yy, class_densities[i],
        levels=levels, colors=[color]
    )

axs[1].set_title("Separate GMM components")
axs[2].set_title("Gaussians fitted to known classes")
axs[2].legend(title="Digit")

for ax in axs:
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()