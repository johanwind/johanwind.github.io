import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0,1,5)   # Training data
y = np.sin(np.pi*2*X)    # Training targets
Z = np.linspace(0,1,100) # Data points to predict

# Kernel method
k = lambda x,y : np.exp(-(x.reshape(-1,1)-y.reshape(1,-1))**2)
K = k(X,X)
alpha = np.linalg.solve(K, y)
kernel_prediction = alpha @ k(X,Z)

# Overparameterized linear regression
features = lambda x : (x.reshape(-1,1) / 10) ** np.arange(10).reshape(1,-1)
X_ = features(X)
beta = X_.transpose() @ np.linalg.solve(X_@X_.transpose(), y)
linear_prediction = features(Z) @ beta

px = 1/plt.rcParams['figure.dpi']  # pixel in inches
plt.figure(figsize=(1080*px, 360*px))
ax = plt.subplot(111)

ax.plot(X, y, 'o', label = 'Data points')
ax.plot(Z, kernel_prediction, '--', label = 'Kernel method')
ax.plot(Z, linear_prediction, '-.', label = 'Linear regression')
ax.plot(Z, np.sin(np.pi*2*Z), label = 'Original function')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('x',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.locator_params(axis='y', nbins=8)

plt.tight_layout()
plt.savefig('../images/kernel_olr.png')

plt.show()
