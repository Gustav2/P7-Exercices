# ###################################
# Group ID : B2-205 (we haven't recieved group numbers yet, thus the group room instead)
# Members : Nikolaj Ask Albertsen, Rui Maria Martins Loureiro, Bjarki Fróðason Í Eyðansstovu,
#  Magnus Vørs Holmsgaard, Gustav Søndergaard Nybro
# Date : 11/9 - 2026
# Lecture: 1 - Introduction to Machine Learning
# Dependencies: scikit-learn, numpy, matplotlib
# Python version: 3.13
# Functionality: This script fits a polynomial regiression model to the sampled data,
#  to show how degree, regularization and sample size affect overfitting.
# ###################################


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# %%
# Load data
work_dir = os.getcwd() # CHANGE THIS WITH YOUR CURRENT DIR!
save_path = work_dir + r'/data/ex01_data.npy'

data = np.load(save_path)
x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]

# %%
# Sample data and plot
np.random.seed(100)

sample_points = 10
sample_idx = np.sort(np.random.choice(len(x1), sample_points))

x1_sub = np.take(x1, sample_idx)
x2_sub = np.take(x2, sample_idx)

plt.figure()
plt.scatter(x1_sub, x2_sub, label='Sampled data')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.show()

# %%
def polyfit(x1, x2, deg, regularization=0, y = None, show_sums_of_squares=False):

    # Generate polynomial features
    poly = PolynomialFeatures(degree=deg)
    x1_poly = poly.fit_transform(x1[:, np.newaxis])

    # Create the Ridge regression model
    ridge_reg = Ridge(alpha=regularization)

    # Fit the model
    ridge_reg.fit(x1_poly, x2)

    # Generate x values for the regression line/curve
    x = np.linspace(x1.min(), x1.max(), 1000)
    X_poly = poly.transform(x[:, np.newaxis])

    # Predict y_hat values
    y_hat = ridge_reg.predict(X_poly) # Use the ridge regression to predict x2 given x1

    # Extract coefficients
    coefs = ridge_reg.intercept_, *ridge_reg.coef_[1:]

    # Here we calculate mean-square-error (MSE)
    MSE = np.mean((x2 - ridge_reg.predict(x1_poly))**2)

    # At last we plot the observation and the regression line/curve
    plt.figure()

    # The following 2 lines is only to show the "squares" to be minimized
    if show_sums_of_squares:
        plt.plot((x1, x1), (x2, ridge_reg.predict(x1_poly)),
                 color='limegreen')

    plt.scatter(x1, x2, label='Observations')
    plt.plot(x, y_hat, '-', label='Polynomial fit: degree %i, reg=%.2g' % (deg, regularization), color='orange')
    if y is not None:
        plt.plot(np.linspace(-5,5,500), y, label='Ground truth', color='red')


    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.ylim([-20, 40])
    plt.legend()
    plt.grid()
    plt.show()

    return coefs, MSE

# %%
# 1) Experiment with different polynomial degrees (without L2 regularization.):
for deg in range(10):
    coefs, mse = polyfit(x1_sub, x2_sub, deg=deg, y=y, show_sums_of_squares=True)
    print('Degree = ', deg, '| MSE = ', round(mse, 3))


# %%
# 2) Explore L2 regularization:
deg = 9
reg = (10, 1, 0.1, 0.01)
for r in reg:
    coefs, mse = polyfit(x1_sub, x2_sub, deg=deg, regularization=r, y=y)
    print('Regularization = ', r, '| MSE = ', round(mse,3))

# %%
# 3) Investigate the effect of sample size.
np.random.seed(100)
sample_points = 100

sample_idx = np.random.choice(len(x1), sample_points, replace=False)

x1_sub_100 = x1[sample_idx]
x2_sub_100 = x2[sample_idx]

deg = 9
reg = 0

coefs, mse = polyfit(x1_sub_100, x2_sub_100, deg=deg, regularization=reg, y=y)
print('Sample size = ', sample_points, '| MSE = ', round(mse, 3))
