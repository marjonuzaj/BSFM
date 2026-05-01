import numpy as np 
import pandas as pd
import sfm
import matplotlib.pyplot as plt

N = 1000
x1 = np.random.normal(0,1,(N,1))
x2 = np.random.normal(0,1,(N,1))
u = np.abs(np.random.normal(0,0.2,(N,1)))
v = np.random.normal(0,0.1,(N,1))
y = 1 + 0.5 *x1 + 0.4 * x2 + v -u
    
df = pd.DataFrame()
df['y'] = y.flatten()
df['x1'] = x1.flatten()
df['x2'] = x2.flatten()

model = sfm.HN(df,'y')
    
model.fit(nsim = 10000, drop= 2000)
print(model.summary())

plt.scatter(model.inef_est, u)
plt.xlabel('Inef Est')
plt.ylabel('Inef Real')
plt.show()

plt.hist(model.inef_est, bins=50, edgecolor='lightgrey')
plt.xlabel('Inef Est')
plt.xlim(0,max(model.inef_est)+0.1)
plt.show()