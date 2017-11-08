
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data_path = '../wsdm_challenge/data/'


# In[3]:


train_file_path = data_path + 'train.csv'


# In[4]:


test_file_path = data_path + 'test.csv'


# In[5]:


members_file_path = data_path + 'members.csv'


# In[6]:


train = pd.read_csv(train_file_path, index_col=None)


# In[7]:


test = pd.read_csv(test_file_path, index_col=None)


# In[8]:


members = pd.read_csv(members_file_path, index_col=None)


# In[14]:


mem_train_test = np.intersect1d(train['msno'].unique(), test['msno'].unique())


# In[15]:


print len(train), len(test), len(mem_train_test)


# In[16]:


shortlisted = np.union1d(train['msno'].unique(), test['msno'].unique())


# In[17]:


print len(shortlisted)


# In[13]:


df = pd.DataFrame(members)


# In[18]:


new_mem = df.loc[df['msno'].isin(shortlisted)]


# In[19]:


print len(new_mem)


# In[20]:


print len(train['msno'].unique())


# In[21]:


print len(test['msno'].unique())


# In[23]:


print (new_mem).head()


# In[24]:


def write_to_csv(dataframe):
    dataframe.to_csv(data_path + 'mem_shortlist.csv', index=False)


# In[25]:


write_to_csv(new_mem)

