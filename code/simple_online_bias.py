import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm  # Optional loading bar
th.manual_seed(0)

d = 256      # Input and output dimension
r = 2        # Rank of ground truth matrix
lr = 2e-8    # Learning rate
yscale = 100 # Scale factor for ground truth

T = th.randn(d,r)@th.randn(r,d) * yscale

W1 = th.randn(d,d, requires_grad=True)
W2 = th.randn(d,d, requires_grad=True)
th.nn.init.xavier_normal_(W1)
th.nn.init.xavier_normal_(W2)

log = []
for i in tqdm(range(d*d//2)):
  a, b = th.randn(d), th.randn(d)
  loss = 0.5*(a@T@b-a@W1@W2@b)**2
  assert(loss == loss)
  loss.backward()
  with th.no_grad():
    for param in [W1,W2]:
      param -= lr * param.grad # Gradient descent
      param.grad[:] = 0
  if i%100 == 0:
    error = (W1@W2-T).norm()/T.norm() # Relative error
    log.append((i,error))

px = 1/plt.rcParams['figure.dpi']  # pixel in inches
plt.figure(figsize=(1080*px, 360*px))
ax = plt.subplot(111)

ax.plot(*zip(*log))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Training samples',fontsize=15)
plt.ylabel('Relative reconstruction error',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('../images/simple_online_bias.png')
plt.show()
