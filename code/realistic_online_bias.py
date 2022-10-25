import matplotlib.pyplot as plt
from tqdm import tqdm # Optional loading bar
import torch as th
nn = th.nn
F = nn.functional
th.manual_seed(0)

device = 'cuda'
#device = 'cpu'

d = 1024      # Input and output dimension
r = 5         # Rank of ground truth matrix
B = 256       # Batch size
L = 10        # Number of layers
yscale = 4e-2 # Scale factor for ground truth

truth = (th.randn(d,r)@th.randn(r,d)).to(device) * yscale

cnn = nn.Sequential(*[x for i in range(L) for x in [nn.Linear(d,d,bias=False), nn.BatchNorm1d(d), nn.ReLU()]][:-2]).to(device)
sgd = th.optim.SGD(cnn.parameters(), lr=2.5e-8, momentum=0.9)

transformer = nn.Sequential(*[x for i in range(L) for x in [nn.Linear(d,d,bias=False), nn.ReLU(), nn.LayerNorm(d)]][:-2]).to(device)
adam = th.optim.Adam(transformer.parameters(), lr=6e-4)

px = 1/plt.rcParams['figure.dpi']  # pixel in inches
plt.figure(figsize=(1080*px, 360*px))
ax = plt.subplot(111)

for net,opt,label in [(cnn,sgd,'CNN setup'), (transformer,adam,'Transformer setup')]:
  log = []
  for bi in tqdm(range(0, d**2//2, B)):
    a = th.randn(B, d, device=device)
    b = th.randn(B, d, device=device)
    y    = th.einsum('bi,bi->b', a@truth, b)
    pred = th.einsum('bi,bi->b', net(a), b)
    loss = th.sum((y-pred)**2)

    log.append((bi, (loss/th.sum(y**2)).item() ))

    opt.zero_grad()
    loss.backward()
    opt.step()

  plt.plot(*zip(*log), label=label)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Training samples',fontsize=15)
plt.ylabel('Relative error',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('../images/realistic_online_bias.png')
plt.show()
