
# coding: utf-8

# In[1]:


from random import shuffle

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding
import pandas as pd
import numpy as np
import scipy.sparse
import tensorflow as tf
import pickle


# In[2]:


members = pd.read_csv('../New_Data/mem_shortlist.csv', index_col=None)
members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members = members.drop(['registration_init_time'], axis=1)
members = members.drop(['expiration_date'], axis=1)

print members.head()
print len(members)


# In[3]:


# Some of the rows corresponding to a column have multiple values separated by '|'
# character. We need to split and separate these multiple values

def get_unique_entities(data, column):
    data[column] = data[column].apply(str)
    return data[column].unique()

def generate_mapper(data, column):
    unique_elements = get_unique_entities(data, column)
    mapper = dict()
    mapper[''] = 0
    for u in unique_elements:
        if u is not '':
            mapper[u] = len(mapper)
    return mapper


# In[7]:


city_mapper = generate_mapper(members, 'city')
msno_mapper = generate_mapper(members, 'msno')
reg_via_mapper = generate_mapper(members, 'registered_via')
reg_year_mapper = generate_mapper(members, 'registration_year')
expiry_year_mapper = generate_mapper(members, 'expiration_year')

mappers = [msno_mapper, city_mapper, reg_via_mapper, reg_year_mapper, expiry_year_mapper]


# In[8]:


with open('msno_mapper.pkl', 'wb') as fw:
    pickle.dump(msno_mapper, fw, protocol=2) #Python 2


# In[9]:


def input_generator(data):
    num_rows = data.shape[0]
    X = np.empty(data.shape[0])
    Y0 = np.empty((data.shape[0], ))
    Y1 = np.empty((data.shape[0], ))
    Y2 = np.empty((data.shape[0], ))
    Y3 = np.empty((data.shape[0], ))

    count = 0
    for row_num, row in data.iterrows():
        X[count] = msno_mapper[row['msno']]
        Y0[count] = city_mapper[row['city']]
        Y1[count] = reg_via_mapper[row['registered_via']]
        Y2[count] = reg_year_mapper[row['registration_year']]
        Y3[count] = expiry_year_mapper[str(row['expiration_year'])]
        count += 1

    return (X, [Y0, Y1, Y2, Y3]) 


# In[10]:


batch_size = 64
num_hidden_units = 50
hidden_activation = 'relu'


# In[15]:


cont = False


# In[19]:


if not cont:
    input_features = Input(shape = (1,))
    embedding = keras.layers.Flatten()(
        Embedding(output_dim = num_hidden_units, input_dim = len(msno_mapper), input_length = 1)(input_features))
    embedding = keras.layers.Activation(hidden_activation)(embedding)
    output_0 = Dense(len(city_mapper), activation='softmax')(embedding)
    output_1 = Dense(len(reg_via_mapper), activation='softmax')(embedding)
    output_2 = Dense(len(reg_year_mapper), activation='softmax')(embedding)
    output_3 = Dense(len(expiry_year_mapper), activation='softmax')(embedding)

    model = keras.models.Model(inputs = [input_features],
                               outputs = [output_0, output_1, output_2, output_3])

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

else:
    model = keras.models.load_model('user_embeddings_100.h5')

X, Ys = input_generator(members)
model.fit(X, Ys, batch_size = batch_size, epochs = 100, verbose=2)
print(model.evaluate(X, Ys))
model.save('user_embeddings_100.h5')


# In[23]:


#get_ipython().system(u'ipython nbconvert --to script user_embeddings.ipynb')

