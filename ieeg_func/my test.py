import matplotlib.pyplot as plt
import numpy as np

s=[]
d=np.arange(0,10,0.1)
s.append(np.sin(2*3*np.pi*d))
s.append(np.sin(2*1*np.pi*d))
s.append(np.sin(2*1*np.pi*d))

fig, ax =plt.subplots(3, 1, figsize=(30, 20))
for i in range(3):
    ax[i].plot(d, s[i])

#fig.savefig('test.png')
plt.show()
