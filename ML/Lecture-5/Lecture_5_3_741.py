# %%
from scipy.io import loadmat
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

# %% [markdown]
# # Exercise 5: Clustering
# This assignment is based on the previously generated 2-dimensional data of the three classes (5, 6 and 8) from the MNIST database of handwritten digits.
#
# First, mix the 2-dimensional data (training data only) by removing the labels and then use one Gaussian mixture model to model them.
#
# Secondly, compare the Gaussian mixture model with the Gaussian models trained in the previous assignment, in terms of mean and variance values as well as through visualisation.

# %% [markdown]
# ## Loading the data and mixing
# First we load the exercise data set, combine the individual training sets into one and shuffle the data to ensure a random shuffle (here with a seed to ensure reproducability).

# %%
data_path = "2D568class.mat"
data = loadmat(data_path)
train5 = data["trn5_2dim"] / 255
train6 = data["trn6_2dim"] / 255
train8 = data["trn8_2dim"] / 255

trainset = np.concatenate([train5, train6, train8])
np.random.seed(0)
np.random.shuffle(trainset)

# %% [markdown]
# ## Creating a Gaussian Mixture model
# First create a Gaussian Mixture Model of the data using sklearn

# %%


# %% [markdown]
# ## Creating Gaussian models
# Following the same approach from the previous exercises we can also estimate Gaussian models for each class

# %%


# %% [markdown]
# ## Comparing means and covariance matrices.
# Let's look at the means and covariance matrices.
#
# First we extract the means and covariances from the GMM.

# %%


# %% [markdown]
# Now we can compare the GMM means and covariances to the Gaussin models estimated for each class individually.

# %% [markdown]
# ### Means

# %%
# for name, mean in {"mean5": mean5, "mean6": mean6, "mean8": mean8,
#                    "mean1_gmm": mean1_gmm, "mean2_gmm": mean2_gmm, "mean3_gmm": mean3_gmm}.items():
#     print(f"{name}: {np.array2string(mean)}")

# %% [markdown]
# ### Covariances

# %%
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# axs[0, 0].matshow(cov5)
# for (i, j), z in np.ndenumerate(cov5):
#     axs[0, 0].text(j, i, f'{z:0.1f}', ha='center', va='center')
# axs[0, 0].set_title("Cov. Class: 5")

# axs[1, 0].matshow(cov1_gmm)
# for (i, j), z in np.ndenumerate(cov1_gmm):
#     axs[1, 0].text(j, i, f'{z:0.1f}', ha='center', va='center')
# axs[1, 0].set_title("Cov. GMM kernel 1")

# axs[0, 1].matshow(cov6)
# for (i, j), z in np.ndenumerate(cov6):
#     axs[0, 1].text(j, i, f'{z:0.1f}', ha='center', va='center')
# axs[0, 1].set_title("Cov. Class: 6")

# axs[1, 1].matshow(cov2_gmm)
# for (i, j), z in np.ndenumerate(cov2_gmm):
#     axs[1, 1].text(j, i, f'{z:0.1f}', ha='center', va='center')
# axs[1, 1].set_title("Cov. GMM kernel 2")

# axs[0, 2].matshow(cov8)
# for (i, j), z in np.ndenumerate(cov8):
#     axs[0, 2].text(j, i, f'{z:0.1f}', ha='center', va='center')
# axs[0, 2].set_title("Cov. Class: 8")

# c = axs[1, 2].matshow(cov1_gmm)
# for (i, j), z in np.ndenumerate(cov3_gmm):
#     axs[1, 2].text(j, i, f'{z:0.1f}', ha='center', va='center')
# axs[1, 2].set_title("Cov. GMM kernel 3")

# %% [markdown]
# What do we see when comparing means and covariances?

# %% [markdown]
# ## Visualizing the models in contourplots.
# Now we would like to visualize our models to compare them.

# %% [markdown]
# We first generate some points to be able to sample from the models.

# %%
# Create points to do a contour a plot

# %% [markdown]
# We can also visualize the separate Gaussian models from the GMM by creating Gassians from the classwise means and covariances of the GMM model.

# %%


# %% [markdown]
# Now we sample from the models using the generated points.

# %%


# %% [markdown]
# The model samples can then be visualized in a contour plot.

# %%
# Plot contours for the GMM, seperated GMM and individual estimated densities

# %%
