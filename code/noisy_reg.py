import torch as th
th.manual_seed(0)

d = 30  # Input dimension
n = 200 # Training examples / testing points

X = th.randn(n*2, d) # Generate random points
y = X[:,0]**2        # Ground truth
X_train, y_train = X[:n,:], y[:n] # Train/test split
X_test,  y_test  = X[n:,:], y[n:]


if 0:
  import matplotlib.pyplot as plt
  px = 1/plt.rcParams['figure.dpi']  # pixel in inches
  plt.figure(figsize=(1080*px, 360*px))
  ax1 = plt.subplot(121)
  ax2 = plt.subplot(122)

  ax1.scatter(X[:,0],y)
  ax2.scatter(X[:,1],y)

  ax1.set_xlabel('Feature #1')
  ax1.set_ylabel('Target')
  ax2.set_xlabel('Feature #2')
  ax2.set_ylabel('Target')
  plt.tight_layout()
  plt.savefig('../images/noisy_reg_data.png')
  plt.show()


def makeModel(seed = 0):
  m = 100 # Number of hidden nodes
  A = th.zeros(d,m, requires_grad=True)
  b = th.zeros(m, requires_grad=True)
  th.manual_seed(seed)
  th.nn.init.xavier_normal_(A)
  parameters = [A,b]
  predict = lambda X : (X@A)**2 @ b
  return parameters, predict


def trainModel(parameters, predict, lr):
  loss = 1e100
  while loss > 1e-4:
    loss = th.mean((predict(X_train)-y_train)**2)
    loss.backward()
    with th.no_grad():
      for param in parameters:
        param -= lr * param.grad # Gradient descent
        param.grad[:] = 0
  return th.mean((predict(X_test)-y_test)**2) # Return test MSE

if 0:
  print('MSE = %.2f'%trainModel(*makeModel(), 0.01))
  print('constant 1 MSE = %.2f'%th.mean((y_test-1)**2))


if 0:
  lr_list = th.arange(0.001,0.15,0.005)
  mse_list = []
  for lr in lr_list:
    mse = trainModel(*makeModel(), lr)
    print("MSE = %.2f"%mse)
    mse_list.append(mse)
  import matplotlib.pyplot as plt
  px = 1/plt.rcParams['figure.dpi']  # pixel in inches
  plt.figure(figsize=(1080*px, 360*px))
  ax = plt.subplot(111)

  ax.plot(lr_list, mse_list)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  plt.xlabel('Learning rate',fontsize=15)
  plt.ylabel('MSE',fontsize=15)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)

  plt.grid(True, axis='y')

  plt.tight_layout()
  plt.savefig('../images/noisy_reg_lr.png')

  plt.show()

def trainModelWithLabelNoise(parameters, predict, lr, steps):
  for _ in range(steps):
    y = y_train + th.randn(y_train.shape)
    loss = th.mean((predict(X_train) - y)**2)
    loss.backward()
    with th.no_grad():
      for param in parameters:
        param -= lr * param.grad # Gradient descent
        param.grad[:] = 0
  return th.mean((predict(X_test)-y_test)**2) # Return test MSE

if 1:
  print('MSE = %.2f'%trainModelWithLabelNoise(*makeModel(), lr = 0.03, steps = 100000))


def makeModelSmallInit(seed = 0):
  m = 100 # Number of hidden nodes
  A = th.zeros(d,m, requires_grad=True)
  b = th.zeros(m, requires_grad=True)
  th.manual_seed(seed)
  th.nn.init.xavier_normal_(A)
  parameters = [A,b]
  predict = lambda X : (X@A)**2 @ b / 100
  return parameters, predict

if 0:
  print('MSE = %.4f'%trainModel(*makeModelSmallInit(), 0.01))


def trainModel(parameters, predict, lr, R = 0):
  train_path, test_path = [], []
  for it in range(100000):
    noise = th.randn(y_train.shape)
    loss = th.mean((predict(X_train)-y_train + noise)**2)
    if R: loss += R()
    loss.backward()
    with th.no_grad():
      for param in parameters:
        param -= lr * param.grad # Gradient descent
        param.grad[:] = 0

    train_loss = (th.mean((predict(X_train)-y_train)**2) / th.mean(y_train**2)).item()
    train_path.append(train_loss)
    test_loss = (th.mean((predict(X_test)-y_test)**2) / th.mean(y_test**2)).item()
    test_path.append(test_loss)
    if it%100 == 0: print(train_loss, test_loss)
  return train_path, test_path


if 0:
  A = th.zeros(d,d, requires_grad=True)
  parameters = [A]
  predict = lambda X : th.einsum('bi,bj,ij->b',X,X,A)
  R = lambda : th.linalg.matrix_norm(A,ord='nuc') / 30
  #R = lambda : th.sum(A**2) / 100
  train_path, test_path = trainModel(parameters, predict, lr=1e-2, R=R)

if 0:
  m = 100 # Number of hidden nodes
  A = th.randn(d,m, requires_grad=True)
  b = th.zeros(m, requires_grad=True)
  th.nn.init.xavier_normal_(A)
  parameters = [A,b]
  predict = lambda X : (X@A)**2 @ b
  train_path, test_path = trainModel(parameters, predict, lr=3e-2)

if 0:
  plt.plot(train_path, label='train')
  plt.plot(test_path, label='test')
  plt.legend()
  plt.show()
