
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re
get_ipython().system(u'pip install --user progressbar2')
import progressbar


# In[3]:

fileName = 'artists_copy_new.csv'
outputFileName = 'artists_copy_new.csv'
col_to_be_modified = 'artist_name'

pattern = '\([a-zA-Z0-9\s]+\)'

# In[5]:

bar = progressbar.ProgressBar()

artists_csv = pd.read_csv(fileName, index_col=None)


# In[ ]:

for i in bar(range(0, len(artists_csv))):
    currArtist = str(artists_csv.iloc[i][col_to_be_modified])
    #Split the str on pattern
    group0 = re.search(pattern, currArtist)
    if (group0 != None):
        newArtist = group0.group(0).replace('(', '').replace(')', '')
        artists_csv.set_value(i, col_to_be_modified, newArtist)
print 'Finito'


# In[ ]:

artists_csv.to_csv(outputFileName, index=False)

