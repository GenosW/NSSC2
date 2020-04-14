#!/usr/bin/env python
# coding: utf-8

# In[28]:


import glob 
import numpy as np
import matplotlib.pyplot as plt
files_250 = glob.glob("data/res_250/*.csv") 
files_2000 = glob.glob("data/res_2000/*.csv")
files_4000 = glob.glob("data/res_4000/*.csv")
files_8000 = glob.glob("data/res_8000/*.csv")


# In[29]:


import pandas as pd
import os
import seaborn as sns
data_250 = pd.DataFrame()
data_2000 = pd.DataFrame()
data_4000 = pd.DataFrame()
data_8000 = pd.DataFrame()

for file in files_250:
    if file.endswith('.csv'):
        df = pd.read_csv(file)
        data_250 = data_250.append(df, ignore_index=True)
        
for file in files_2000:
    if file.endswith('.csv'):
        df = pd.read_csv(file)
        data_2000 = data_2000.append(df, ignore_index=True)
        
for file in files_4000:
    if file.endswith('.csv'):
        df = pd.read_csv(file)
        data_4000 = data_4000.append(df, ignore_index=True)
        
for file in files_8000:
    if file.endswith('.csv'):
        df = pd.read_csv(file)
        data_8000 = data_8000.append(df, ignore_index=True)


# In[30]:


d = np.arange(1,len(data_250)+1)
data_250
print(len(data_250))


# In[32]:


plt.figure()
sns.lineplot(y="runtime/iter", x=d, data=data_250)
sns.lineplot(y="runtime/iter", x=d, data=data_2000)
sns.lineplot(y="runtime/iter", x=d, data=data_4000)
sns.lineplot(y="runtime/iter", x=d, data=data_8000)
plt.savefig('time_plot.png')


# In[ ]:


import pandas as pd
df = pd.DataFrame
df = pd.read_csv("localDom.csv")


# In[ ]:


df


# In[ ]:




