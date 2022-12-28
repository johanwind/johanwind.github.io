import matplotlib.pyplot as plt
import numpy as np
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
plt.figure(figsize=(1080*px, 360*px))
ax = plt.subplot(111)

EPOCHS, BATCH_SIZE = 22, 512
epoch_list = np.arange(0,EPOCHS,.1)
lr_knots = [0, EPOCHS/5, EPOCHS]
lr_vals  = [0.1/BATCH_SIZE, 0.6/BATCH_SIZE, 0]

ax.plot(epoch_list, np.interp(epoch_list, lr_knots, lr_vals))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Epoch',fontsize=15)
plt.ylabel('lr',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('/home/johan/Documents/goodminima/johanwind.github.io/images/cifar94_lr.png')

plt.show()

