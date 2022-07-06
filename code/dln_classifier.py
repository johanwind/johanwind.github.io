import torch as th
import torch.nn.functional as F
th.manual_seed(0)
import numpy as np

d = 10  # Input dimension
k = 10  # Classes
n = 100 # Training examples / testing points

X = th.randn(n*2, d) # Generate random points
y = ((th.atan2(X[:,0],X[:,1])/np.pi+1)/2*k).long() # Classify points
X_train, y_train = X[:n,:], y[:n] # Train/test split
X_test,  y_test  = X[n:,:], y[n:]

if 1:
  import matplotlib.pyplot as plt
  px = 1/plt.rcParams['figure.dpi']  # pixel in inches
  plt.figure(figsize=(1080*px, 360*px))
  ax1 = plt.subplot(121)
  ax2 = plt.subplot(122)
  for c in range(k):
    ax1.scatter(X[y==c][:,0],X[y==c][:,1])
    ax2.scatter(X[y==c][:,2],X[y==c][:,3])

  ax1.set_xlabel('Feature #1')
  ax1.set_ylabel('Feature #2')
  ax2.set_xlabel('Feature #3')
  ax2.set_ylabel('Feature #4')
  ax1.set_aspect('equal')
  ax2.set_aspect('equal')
  plt.tight_layout()
  plt.savefig('../images/dln_classifier_data.png')
  plt.show()


def trainModel(parameters, predict, lr):
  loss = 1e100
  while loss > 0.01: # Optimize until mean cross entropy loss is <= 0.01
    loss = F.cross_entropy(predict(X_train), y_train)
    loss.backward()
    with th.no_grad():
      for param in parameters:
        param -= lr * param.grad # Gradient descent
        param.grad[:] = 0
  return th.sum(th.argmax(predict(X_test), dim=1) == y_test).item() # Return test accuracy

if 0:
  W = th.zeros(d,k, requires_grad=True)
  parameters = [W]
  predict = lambda X : X@W
  print("Test accurracy:", trainModel(parameters, predict, lr=10), '%')
  exit()

if 0:
  scores = []
  for _ in range(10):
    A = th.zeros(d,k, requires_grad=True)
    B = th.zeros(k,k, requires_grad=True)
    th.nn.init.xavier_normal_(A)
    th.nn.init.xavier_normal_(B)
    parameters = [A,B]
    predict = lambda X : X@A@B
    scores.append(trainModel(parameters, predict, lr=1))
  print("Test accurracy: %.1f ± %.1f %%"%(np.mean(scores), np.std(scores)/len(scores)**.5))
  exit()

if 0:
  import matplotlib.pyplot as plt

  L_list = [1,2,3,4,5,6]
  acc_list = []
  std_list = []
  for L in L_list:
    scores = []
    for _ in range(10):
      layers = []
      for l in range(L):
        layers.append( th.zeros(d if l==0 else k, k, requires_grad=True) )
        th.nn.init.xavier_normal_(layers[l])

      def predict(X):
        product = X
        for layer in layers: product @= layer
        return product

      if L == 1: lr = 10
      elif L == 2: lr = 1
      else: lr = 3e-2
      scores.append(trainModel(layers, predict, lr))
    acc_list.append(np.mean(scores))
    std_list.append(np.std(scores)/len(scores)**.5)
    print("Test accurracy: %.1f ± %.1f %%"%(acc_list[-1], std_list[-1]))

  px = 1/plt.rcParams['figure.dpi']  # pixel in inches
  plt.figure(figsize=(1080*px, 360*px))
  ax = plt.subplot(111)
  ax.errorbar(L_list, acc_list, std_list)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  plt.xlabel('Layers',fontsize=15)
  plt.ylabel('Test Accuracy (%)',fontsize=15)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)

  plt.grid(True, axis='y')

  plt.tight_layout()
  plt.savefig('../images/dln_classifier_depth.png')

  plt.show()

if 0:
  scores = []
  for _ in range(10):
    A = th.zeros(d,2, requires_grad=True)
    B = th.zeros(2,k, requires_grad=True)
    th.nn.init.xavier_normal_(A)
    th.nn.init.xavier_normal_(B)
    parameters = [A,B]
    predict = lambda X : X@A@B
    scores.append(trainModel(parameters, predict, lr=1))
  print("Test accurracy: %.1f ± %.1f %%"%(np.mean(scores), np.std(scores)/len(scores)**.5))


if 0:
  scores = []
  for _ in range(10):
    A = th.zeros(d,k, requires_grad=True)
    B = th.zeros(k,k, requires_grad=True)
    C = th.zeros(k,k, requires_grad=True)
    th.nn.init.xavier_normal_(A)
    th.nn.init.xavier_normal_(B)
    th.nn.init.xavier_normal_(C)
    parameters = [A,B,C]
    predict = lambda X : X@A@B@C / 1e4
    scores.append(trainModel(parameters, predict, lr=100))
  print("Test accurracy: %.1f ± %.1f %%"%(np.mean(scores), np.std(scores)/len(scores)**.5))


if 0:
  import matplotlib.pyplot as plt

  lr = 100
  eps = 1e-4
  #lr = 1e-1
  #eps = 1.

  A = th.zeros(d,k, requires_grad=True)
  B = th.zeros(k,k, requires_grad=True)
  C = th.zeros(k,k, requires_grad=True)
  th.nn.init.xavier_normal_(A)
  th.nn.init.xavier_normal_(B)
  th.nn.init.xavier_normal_(C)
  parameters = [A,B,C]
  predict = lambda X : X@A@B@C * eps

  sigma_list = [[] for i in range(5)]
  loss = 1e100
  while loss > 0.01: # Optimize until mean cross entropy loss is <= 0.01
    loss = F.cross_entropy(predict(X_train), y_train)
    loss.backward()
    with th.no_grad():
      for param in parameters:
        param -= lr * param.grad # Gradient descent
        param.grad[:] = 0
    sigmas = th.linalg.svdvals(A@B@C * eps)
    for i in range(len(sigma_list)):
      sigma_list[i].append(sigmas[i])


  px = 1/plt.rcParams['figure.dpi']  # pixel in inches
  plt.figure(figsize=(1080*px, 360*px))
  ax = plt.subplot(111)

  for i in range(len(sigma_list)):
    ax.plot(sigma_list[i], label="Singular value #"+str(i+1))

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  plt.xlabel('Training iteration',fontsize=15)
  plt.ylabel('Singular value size',fontsize=15)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  plt.legend(fontsize=15)

  plt.tight_layout()
  plt.savefig('../images/dln_classifier_singular_small.png')

  plt.show()
