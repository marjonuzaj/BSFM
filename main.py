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

model1 = sfm.HN(df,'y')
model2 = sfm.EXP(df,'y')

model1.fit(nsim = 10000, drop= 2000)
print(model1.summary())

model2.fit(nsim = 10000, drop= 2000)
print(model2.summary())

plt.scatter(model1.inef_est, model2.inef_est)
plt.xlabel('Inef Est Exp')
plt.ylabel('Inef Est HN')
plt.show()

plt.hist(model1.inef_est, bins=50, edgecolor='lightgrey', label='HN')
plt.hist(model2.inef_est, bins=50, edgecolor='lightgrey', label='Exp')
plt.xlabel('Inef Est')
plt.xlim(0,max(model1.inef_est)+0.1)
plt.legend()
plt.show()