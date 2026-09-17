# ###################################
# Group ID : 741
# Members : Nikolaj Ask Albertsen, Rui Maria Martins Loureiro, Bjarki Fróðason Í Eyðansstovu,
#  Magnus Vørs Holmsgaard, Gustav Søndergaard Nybro
# Date : 17/9 - 2026
# Lecture: 2/3 - Bayesian decision theory / Parametric and nonparametric methods
# Dependencies: numpy
# Python version: 3.13
# Functionality: Gaussian Bayes classifier
# ###################################
#
#
# # Exercise
# ## You are given, as the train data, trn_x and trn_y along with their class labels trn_x_class and trn_y_class. The task is to classify the following TEST data.
#

# First we load the data from the text files

# In[1]:


import numpy as np

# Train data
train_x = np.loadtxt("data/trn_x.txt")
train_x_label = np.loadtxt("data/trn_x_class.txt")

train_y = np.loadtxt("data/trn_y.txt")
train_y_label = np.loadtxt("data/trn_y_class.txt")

train_mean = np.mean(train_x, axis= 0)
train_var = np.var(train_x, axis= 0)

# Test data
test_x = np.loadtxt("data/tst_x.txt")
test_x_label = np.loadtxt("data/tst_x_class.txt")

test_y = np.loadtxt("data/tst_y.txt")
test_y_label = np.loadtxt("data/tst_y_class.txt")

test_y_126 = np.loadtxt("data/tst_y_126.txt")
test_y_126_label = np.loadtxt("data/tst_y_126_class.txt")

test_xy = np.loadtxt("data/tst_xy.txt")
test_xy_label = np.loadtxt("data/tst_xy_class.txt")

test_xy_126 = np.loadtxt("data/tst_xy_126.txt")
test_xy_126_label = np.loadtxt("data/tst_xy_126_class.txt")


# Looking at the data we see that our input features is 2-dimensional, i.e., it has two values per data point.
# Furthermore, x has label 1 and y has label 2.

# Let's visualize the training data by plotting a 2D scatter plot and corresponding Gaussians for class x and class y

# In[ ]:


# Hint: look at: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html


# ### (a) classify instances in tst_xy, and use the corresponding label file tst_xy_class to calculate the accuracy;

# First we compute the statistics of x and y (use np.mean and np.cov)

# In[2]:


# x statistics
train_x_mean = np.mean(train_x, axis=0)
train_x_cov = np.cov(train_x, rowvar=False)

# y statistics
train_y_mean = np.mean(train_y, axis=0)
train_y_cov = np.cov(train_y, rowvar=False)

# priors
prior_x = train_x.shape[0] / (train_x.shape[0] + train_y.shape[0])
prior_y = train_y.shape[0] / (train_x.shape[0] + train_y.shape[0])


# Now we need to a function for computing the likelihood of x and y given our test data.

# In[3]:


# Define likelihood function
# Implement your own or look on stack overflow if you are lazy - most important thing is that you understand what is going on

def likelihood(data, mean, cov):
    d = mean.shape[0]
    diff = data - mean
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    likelihood_value = (1.0 / np.sqrt((2 * np.pi) ** d * cov_det)) \
        * np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
    return likelihood_value


# To classify the test data we compute the likelihood of it being class x and class y

# In[4]:


# Compute likelihood of x and y
likelihood_x = likelihood(test_xy, train_x_mean, train_x_cov)
likelihood_y = likelihood(test_xy, train_y_mean, train_y_cov)


# We compute the posterior probability by taking the priors into account

# In[ ]:


# Compute posteriors from likelihood and prior
posterior_x = likelihood_x * prior_x
posterior_y = likelihood_y * prior_y


# Now choose to classify our test data as belonging to the class with the highest posterior probability

# In[5]:


# Remember that labels for x and y are are 1 and 2 respectively
classification = np.where(posterior_x > posterior_y, 1, 2)


# We can compute the accuracy of our classifications by taking the sum of correct predictions and divide by the total number of predictions

# In[6]:


accuracy_xy = np.mean(classification == test_xy_label)
print(f"Accuracy on tst_xy: {accuracy_xy*100:.2f}%")


# ### (b) classify instances in tst_xy_126 by assuming a uniform prior over the space of hypotheses, and use the corresponding label file tst_xy_126_class to calculate the accuracy;

# First we define our prior probabilities

# In[7]:


prior_x_uniform = 0.5
prior_y_uniform = 0.5


# We can now compute posteriors knowing that the posterior probability is simply the prior, p(C), multiplied by the likelihood p(x, C).

# In[8]:


likelihood_x_uniform = likelihood(test_xy_126, train_x_mean, train_x_cov)
likelihood_y_uniform = likelihood(test_xy_126, train_y_mean, train_y_cov)

posterior_x_uniform = likelihood_x_uniform * prior_x_uniform
posterior_y_uniform = likelihood_y_uniform * prior_y_uniform


# Now that we have posteriors for both x and y we can classify the test data and compute the accuracy

# In[3]:


classification_uniform = np.where(posterior_x_uniform > posterior_y_uniform, 1, 2)

accuracy_xy_126_uniform = np.mean(classification_uniform == test_xy_126_label)
print(f"Accuracy using uniform prior {accuracy_xy_126_uniform*100:.2f}%")


# ### (c) classify instances in tst_xy_126 by assuming a prior probability of 0.9 for Class x and 0.1 for Class y, and use the corresponding label file tst_xy_126_class to calculate the accuracy; compare the results with those of (b).

# Here we simply follow the procedure of (b), however, this time with updated priors

# In[1]:


prior_x_non_uniform = 0.9
prior_y_non_uniform = 0.1

likelihood_x_non_uniform = likelihood(test_xy_126, train_x_mean, train_x_cov)
likelihood_y_non_uniform = likelihood(test_xy_126, train_y_mean, train_y_cov)
posterior_x_non_uniform = likelihood_x_non_uniform * prior_x_non_uniform
posterior_y_non_uniform = likelihood_y_non_uniform * prior_y_non_uniform

classification_non_uniform = np.where(posterior_x_non_uniform > posterior_y_non_uniform, 1, 2)

accuracy_xy_126_non_uniform = np.mean(classification_non_uniform == test_xy_126_label)

print(f"Accuracy using non-uniform prior {accuracy_xy_126_non_uniform*100:.2f}%")


# Comparing the accuracy using uniform prior and non-uniform priors we see that using prior information about the data distribution improves classifcation accuracy by ?%.

# In[2]:


improvement = (accuracy_xy_126_non_uniform / accuracy_xy_126_uniform) - 1
print(f"Absolute improvement in accuracy {improvement*100:.2f}%")


# In[ ]:
