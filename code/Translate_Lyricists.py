
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from langid.langid import LanguageIdentifier, model
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
from googletrans import Translator
translator = Translator()
import time
import sys

default_stdout = sys.stdout
default_stderr = sys.stderr

reload(sys)
sys.setdefaultencoding('utf8')

sys.stdout = default_stdout
sys.stderr = default_stderr


# In[2]:

data_path = '../WSDM/'


# In[3]:

print ('Loading the lyricists csv')


# In[4]:

lyricist_csv = data_path + 'lyricist.csv'


# In[5]:

lyricists = pd.read_csv(lyricist_csv, index_col=None, usecols=['song_id', 'lyricist'])


# In[6]:

lyricists['translated_names'] = lyricists['lyricist']
# Adding a new column to the existing table
dict_names = {}


# In[7]:

def translate_sent(sent,i, dest='en'):
    count = 0
    while count <= 5 :
        try :
            y = translator.translate(sent.decode('utf8'), dest='en')
            text = y.text
            dict_names[sent] = text
            time.sleep(1)
            return text
        except Exception as e:
            print 'Caught exception, retrying for ', sent, i
            time.sleep(pow(2, count))
        count += 1
    return sent


# In[ ]:

print 'Starting translation'
for i in range(0,len(lyricists)):
    currName = str(lyricists.iloc[i]['lyricist'])
    lang, pred = identifier.classify(currName)  # Using LangId
    #     print lang, pred, artists.iloc[i]['artist_name'], i
    if (lang != 'en'):
        if currName in dict_names:
            text = dict_names[currName]
        else:
            text = translate_sent(currName, i, dest='en')
        print text, i
        lyricists.set_value(i, 'translated_names', text)
print 'Translation finished'


# In[ ]:

def write_to_csv(dataframe):
    dataframe.to_csv(data_path + 'tr_lyricist.csv', index=False)


# In[ ]:

write_to_csv(lyricists)


# In[ ]:

print 'Done'


# In[ ]:



