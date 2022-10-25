from tqdm import tqdm
import matplotlib.pyplot as plt
import torch as th
nn = th.nn
F = nn.functional

device = 'cuda'
#device = 'cpu'

# Input and output dimension
d = 1024
# Rank of ground truth matrix
r = 2
# Batch size
B = 64

dofs = d*r*2-r*r
prec = 1e-1

print('dofs', dofs, 'd*d', d*d)
yscale = 3

warmup = 1e-9 #d*r*10

# Number of layers
L = 10

DLN   = lambda : [nn.Linear(d,d,bias=False) for i in range(L)]
MLP   = lambda : [x for i in range(L) for x in [nn.Linear(d,d,bias=False), nn.ReLU()]][:-1]
BNnet = lambda : [x for i in range(L) for x in [nn.Linear(d,d,bias=False), nn.BatchNorm1d(d), nn.ReLU()]][:-2]
LNnet = lambda : [x for i in range(L) for x in [nn.Linear(d,d,bias=False), nn.ReLU(), nn.LayerNorm(d)]][:-2]

truth = (th.randn(d,r)@th.randn(r,d)).to(device)

if 0:
  need = 0
  while 1:
    need += d
    X0 = th.randn(need, d, device=device)
    X1 = th.randn(need, d, device=device)
    y = th.einsum('bi,bi->b', X0@truth, X1)
    y = th.linalg.solve((X0@X0.t())*(X1@X1.t()), y)
    cur = X0.t()*y@X1
    #y = th.einsum('bi,bi->b', X0@(truth-cur), X1)
    print((truth-cur).norm()/truth.norm())
    if (truth-cur).norm() < prec * truth.norm(): break
  assert((truth-cur).norm() < prec * truth.norm()) # Check for NaN
  print("L2 minimization takes", need/dofs, ' / ', d*d/dofs)


net = nn.Sequential(*BNnet()).to(device)
#net = nn.Sequential(*DLN()).to(device)

def sample(B):
  X0 = th.randn(B, d, device=device)
  X1 = th.randn(B, d, device=device)
  y = th.einsum('bi,bi->b', X0@truth, X1)
  pred = th.einsum('bi,bi->b', net(X0), X1)
  return y, pred

if 1: # Rescale truth and lr
  scaleB = B
  y,pred = sample(scaleB)
  rescale = (yscale*pred.norm()/y.norm()).item()
  truth *= rescale
  y *= rescale
  loss = th.sum((y-pred)**2)
  loss.backward()
  g2 = sum(th.sum(p.grad**2) for p in net.parameters())
  lr = loss.item()/g2.item()

lr *= 5
opt = th.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

#lr = 23e-4
#opt = th.optim.Adam(net.parameters(), lr=lr, betas=(0.9,0.999))

def train():
  loss_log = []
  for bi in tqdm(range(0, dofs*100, B)):
    y,pred = sample(B)
    loss = th.sum((y-pred)**2)

    rel = (loss/th.sum(y**2)).item()
    loss_log.append(rel)
    if rel < prec or rel > 100: break

    opt.zero_grad()
    loss.backward()
    opt.param_groups[0]['lr'] = lr*min(1,(bi+B)/warmup)
    opt.step()
  return loss_log

loss_log = train()
print(loss_log[-1], len(loss_log)*B / dofs, len(loss_log)*B/(d*d))
plt.plot(range(0,len(loss_log)*B,B), loss_log)
plt.ylim([0,2])
plt.show()

