
# coding: utf-8

# In[1]:


''' This notebook contains the DNN model for song_embeddings + filtered users (no embeddings for users)'''


# In[2]:


import numpy as np
import math
import pandas as pd

from tqdm import tqdm
import keras
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten
from keras.callbacks import ModelCheckpoint
import cPickle as pickle


# In[3]:


def load_song_embeddings():
    '''This method loads the song embeddings model into memory'''
    model = keras.models.load_model('../New_Data/model_song_embeddings/songs_embeddings_100.h5')
    return model


# In[4]:


def load_song_pickle():
    '''This method loads the song ids mapped to their one hot encoding. This will be used to retrieve their embeddings from the song model
    '''
    song_mapper = pickle.load(open('../New_Data/model_song_embeddings/song_mapper_py2.pkl', 'r'))
    return song_mapper


# In[64]:


def load_dataset():
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    return train, test


# In[6]:


train, test = load_dataset()


# In[7]:


print train.shape


# In[109]:


# song_weights = song_model[3]


# In[120]:


#### Get the intermediate layer for songs ####


# In[8]:


song_orig_model = load_song_embeddings()


# In[56]:


song_pickle = load_song_pickle()


# In[10]:


song_orig_embedding_model = Model(inputs=song_orig_model.input,
                                 outputs=song_orig_model.get_layer('embedding_1').output)


# In[11]:


print len(song_pickle)


# In[87]:


unique_songs = train.song_id.unique()
len(unique_songs)

unique_songs_tr = pd.DataFrame(unique_songs, columns=['song_id'], index=None)
print len(unique_songs_tr)


# In[88]:


count = 0
for keys in tqdm(unique_songs):
    if keys not in song_pickle:
        count += 1
        train = train[train.song_id != keys]
print count 

song_weights = song_orig_embedding_model.predict(train.apply(lambda row: song_pickle[row.song_id], axis=1))


# In[89]:


print len(song_weights)


# In[90]:


print song_weights.shape

song_weights = song_weights.reshape(song_weights.shape[0], song_weights.shape[2])
print song_weights.shape


# In[16]:


### Now let's get the user embeddings
def load_user_model():
    model = keras.models.load_model('../New_Data/model_user_embeddings/weights.19.hdf5')
    return model

def load_user_pickle():
    user_mapper = pickle.load(open('../New_Data/model_user_embeddings/msno_mapper_py2.pkl', 'r'))
    return user_mapper


# In[17]:


user_model = load_user_model()
user_mapper = load_user_pickle()


# In[18]:


user_model.summary()


# In[21]:


# user_embeddings_layer = Model(inputs=user_model.input,
                                 #outputs=user_model.get_layer('dense_1').output) 


# In[23]:


# user_weights = user_embeddings_layer.predict(train.apply(lambda row: user_mapper[row.msno], axis=1))


# In[277]:


### Now that we have the song_embeddings, let's just try to train the model ### 


# In[17]:


input_song_ids = Input(shape=(1,))


# In[19]:


def get_unique_entities(data, column):
    data[column] = data[column].apply(str)
    print len(data[column].unique())
    return data[column].unique()

def generate_mapper(data, column):
    unique_elements = get_unique_entities(data, column)
    mapper = dict()
    for u in unique_elements:
        if u is not '':
            mapper[u] = len(mapper)
    return mapper


# In[80]:


song_mapper = generate_mapper(train, 'song_id')

print len(song_mapper)


# In[57]:


print len(song_pickle)

maxVal  = max(song_pickle.values())
print maxVal


# In[210]:


input_song_ids_layer = Input(shape=(100,))


# In[211]:


# song_embeds = Embedding(output_dim=song_weights.shape[1], input_length=1, input_dim=len(song_mapper), weights=[song_weights], trainable=False)(input_song_ids_layer)


# In[212]:


# flatten_embeds = Flatten()(song_embeds)
intermediate_0 = Dense(25)(input_song_ids_layer)
output_0 = Dense(1, activation='sigmoid')(intermediate_0)

print output_0.shape


# In[213]:


dnn_model = keras.models.Model(inputs = [input_song_ids_layer],
                               outputs = [output_0])


# In[214]:


dnn_model.compile(optimizer=keras.optimizers.Adam(lr=1), loss='binary_crossentropy', metrics=['accuracy'])


# In[215]:


print(dnn_model.summary())


# In[166]:


from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

#SVG(model_to_dot(dnn_model).create(prog='dot', format='svg'))


# In[174]:


print song_weights.shape

print song_mapper[song_mapper.keys()[0]]

print song_weights[317,]
print song_weights[318, ]


# In[191]:


output_mapper = generate_mapper(train, 'target')
song_mapper = generate_mapper(train, 'song_id')

def input_generator(data):
    num_rows = data.shape[0]
    X = np.zeros((data.shape[0], 100), dtype='float32')
    Y = np.empty(data.shape[0])

    count = 0
    for row_num, row in data.iterrows():
        X[count,] = song_weights[song_mapper[row['song_id']]]
        Y[count] = output_mapper[row['target']]
        count += 1
    return (X, Y) 


# In[192]:


X, Y = input_generator(train)
batch_size = 128
# print Y[0:10]
# print X[0:10]


# In[193]:


def get_weights(model):
    for layer in model.layers[3:]:
        print layer.name
        print layer.get_weights()


# In[194]:


# get_weights(dnn_model)


# In[238]:


dnn_model.compile(optimizer=keras.optimizers.Adam(lr=1e-2), loss='binary_crossentropy', metrics=['accuracy'])
dnn_model.fit(x=X, y=Y, batch_size = 128, epochs = 20, verbose=2, validation_split=0.3)


# In[173]:


# get_weights(dnn_model)


# In[231]:


# dnn_model.predict(X)

