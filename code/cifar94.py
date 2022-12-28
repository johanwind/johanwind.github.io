import torch, torchvision, sys, time, numpy
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda")
dtype = torch.float16

EPOCHS = 24
BATCH_SIZE = 512
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4*BATCH_SIZE
lr_knots = [0, EPOCHS/5, EPOCHS]
lr_vals  = [0.1/BATCH_SIZE, 0.6/BATCH_SIZE, 0]
W,H = 32,32
CUTSIZE = 8
CROPSIZE = 4

class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    dims = [3,64,128,128,128,256,512,512,512]
    seq = []
    for i in range(len(dims)-1):
      c_in,c_out = dims[i],dims[i+1]
      seq.append( nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False) )
      if c_out == c_in * 2:
        seq.append( nn.MaxPool2d(2) )
      seq.append( nn.BatchNorm2d(c_out) )
      seq.append( nn.CELU(alpha=0.075) )
    self.seq = nn.Sequential(*seq, nn.MaxPool2d(4), nn.Flatten(), nn.Linear(dims[-1], 10, bias=False))

  def forward(self, x, y):
    x = self.seq(x) / 8
    return F.cross_entropy(x, y, reduction='none', label_smoothing=0.2), (x.argmax(dim=1) == y)*100

def loadCIFAR10(device):
  train = torchvision.datasets.CIFAR10(root="./data", train=True, download=False)
  test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=False)
  ret = [torch.tensor(i, device=device) for i in (train.data, train.targets, test.data, test.targets)]
  std, mean = torch.std_mean(ret[0].float(),dim=(0,1,2),unbiased=True,keepdim=True)
  for i in [0,2]: ret[i] = ((ret[i]-mean)/std).to(dtype).permute(0,3,1,2)
  return ret

def getBatches(X,y, istrain):
  if istrain:
    perm = torch.randperm(len(X), device=device)
    X,y = X[perm],y[perm]

    Crop = ([(y0,x0) for x0 in range(CROPSIZE+1) for y0 in range(CROPSIZE+1)], 
        lambda img, y0, x0 : nn.ReflectionPad2d(CROPSIZE)(img)[..., y0:y0+H, x0:x0+W])
    FlipLR = ([(True,),(False,)], 
        lambda img, choice : torch.flip(img,[-1]) if choice else img)
    def cutout(img,y0,x0):
      img[..., y0:y0+CUTSIZE, x0:x0+CUTSIZE] = 0
      return img
    Cutout = ([(y0,x0) for x0 in range(W+1-CUTSIZE) for y0 in range(H+1-CUTSIZE)], cutout)

    for options, transform in (Crop, FlipLR, Cutout):
      optioni = torch.randint(len(options),(len(X),), device=device)
      for i in range(len(options)):
        X[optioni==i] = transform(X[optioni==i], *options[i])

  return ((X[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]) for i in range(0,len(X) - istrain*(len(X)%BATCH_SIZE),BATCH_SIZE))


X_train, y_train, X_test, y_test = loadCIFAR10(device)

model = CNN().to(dtype).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)

training_time = stepi = 0
for epoch in range(EPOCHS):
  start_time = time.perf_counter()
  train_losses, train_accs = [], []
  model.train()
  for X,y in getBatches(X_train,y_train, True):
    stepi += 1
    opt.param_groups[0]['lr'] = numpy.interp([stepi/(len(X_train)//BATCH_SIZE)], lr_knots, lr_vals)[0]

    loss,acc = model(X,y)
    model.zero_grad()
    loss.sum().backward()
    opt.step()
    train_losses.append(loss.detach())
    train_accs.append(acc.detach())
  training_time += time.perf_counter()-start_time

  model.eval()
  with torch.no_grad():
    test_accs = [model(X,y)[1].detach() for X,y in getBatches(X_test,y_test, False)]

  summary = lambda l : torch.mean(torch.cat(l).float())
  print(f'epoch % 2d  train loss %.3f  train acc %.2f  test acc %.2f  training time %.2f'%(epoch+1, summary(train_losses), summary(train_accs), summary(test_accs), training_time))
