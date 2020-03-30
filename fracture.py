#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import pandas as pd
from core.dataloader import DataLoader_for_training, DataLoader_for_predict
from core.model import *
'''import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.models import load_model
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras import initializers'''
from keras.models import load_model
from scipy import misc


# In[2]:


'''for i in range(6):
    original_filename ='data/training_data/original_'+str(i+1)+'.tif'
    labled_filename ='data/training_data/labled_'+str(i+1)+'.tif'
    data = DataLoader_for_training(original_filename,labled_filename,6,6)
    train_x, train_y = data.generate_training_data()
    pd.DataFrame(train_x).to_csv('data/training_data/original_all_6_6.csv',header = None, index = 0, mode = 'a')
    pd.DataFrame(train_y).to_csv('data/training_data/labled_all_6_6.csv',header = None, index = 0, mode = 'a')
    print('data/training_data/original_'+str(i+1)+'.tif' + ' have successfull generated')'''


# In[3]:


train_x = pd.read_csv('data/training_data/original_all_6_6.csv',header = None).values
train_y = pd.read_csv('data/training_data/labled_all_6_6.csv',header = None).values
#test_x = pd.read_csv('data/test1_4_4.csv',header = None).values


# In[4]:


model = model_generate(train_x)
model.fit(train_x,train_y)


# In[5]:


model.save('saved_model/model.h5')
model = keras.models.load_model('saved_model/model.h5')


# In[6]:


predict_data = DataLoader_for_predict('data/test1.tif',6,6)
test_y = predict_data.generate_predict_lable(model)


# In[7]:


misc.imsave('images/test_all_6_6_epoch=1.tif', 1-test_y)


# In[ ]:


for i in range(10):
    model.fit(train_x,train_y)
    predict_data = DataLoader_for_predict('data/test1.tif',6,6)
    test_y = predict_data.generate_predict_lable(model)
    misc.imsave('images/test_all_6_6_epoch='+str(i+2)+'.tif', 1-test_y)


# In[ ]:




