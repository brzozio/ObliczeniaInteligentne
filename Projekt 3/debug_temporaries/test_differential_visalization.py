import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datasets_get import mnist_extr_diff
import matplotlib as plt

mnist = mnist_extr_diff('cpu', False, "test")
data = mnist.data.detach().numpy()
targets = mnist.targets.detach().numpy()

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 4, figure=fig) 
# image
ax1 = fig.add_subplot(gs[0:3, 0:3])
ax1.plot([1, 2, 3], [4, 5, 6])
# vertical axis - 28:56
ax2 = fig.add_subplot(gs[0:3, 3])
ax2.plot([1, 2, 3], [6, 5, 4])
# horizontal axis = 0:28
ax3 = fig.add_subplot(gs[3, 0:3])
ax3.plot([1, 2, 3], [7, 8, 9])

ax1.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
ax1.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
ax2.tick_params(axis='y',which='both',left=False,right=True,labelleft=False)
ax3.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
ax3.yaxis.tick_right()

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
