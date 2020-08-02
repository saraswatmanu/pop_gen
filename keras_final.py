import numpy as np
import sys
import numpy as np
from keras.layers import Input, Dense,Subtract
from keras.models import Model
import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense, Dropout,concatenate
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D,SeparableConv1D,Flatten
import pickle
from random import sample,shuffle
from keras import regularizers
from keras import optimizers
from keras import callbacks
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Activation
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.layers import *
import random
import math
import time
from keras.layers.advanced_activations import LeakyReLU
from math import log
import keras.backend as K
from keras.layers import Lambda



class TimeHistory(callbacks.Callback):
    def __init__(self):
        self.times=[]
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)



# In[7]:


#Parameters of the model:Lambda parameter

gen_len=int((1.0)*780)
phy_len=int((1.0)*780)




neurons_1=1000
neurons_2=100
neurons_3=100
dropout_1=0.6
dropout_2=0.1
learning_rate=0.001
batch_size=512

# In[9]:


test_data=[]
test_param=[]
i=0
with open('test_data.txt','r') as f:
    for line in f:
        
        i+=1
        
        if i>700000:
            break
        
        
        
        
        
        
        line=line.strip()
        line=line.split()
        if len(line)>1:
            params=line[1:4]
            params=[float(x) for x in params]
            genetic_dis=line[5:785]
            genetic_dis=[float(x) for x in genetic_dis]

            phy_dis=line[786:1566]
            phy_dis=[float(x) for x in phy_dis]

            phy_dis_sorted=[y for x, y in sorted(zip(genetic_dis,phy_dis))]
            genetic_dis_sorted=[x for x, y in sorted(zip(genetic_dis,phy_dis))]
            feature=genetic_dis_sorted[:gen_len] + phy_dis_sorted[:phy_len]
            feature=np.asarray(feature)
            print(len(feature))
            test_data.append(feature)
            test_param.append(params[2])

test_data=np.array(test_data)
test_param=np.array(test_param)
        
        


# In[10]:


#Loading Data from file : data.txt

# order: lambda, mu , radius

data=[]
parameter=[]

i=0

with open('data.txt','r') as f:
    for line in f:
        
        i+=1
        if i>800000:
            break
        
        
        
        
        
        
        
        
        
        line=line.strip()
        line=line.split()
        params=line[1:4]
        params=[float(x) for x in params]
        genetic_dis=line[5:785]
        genetic_dis=[float(x) for x in genetic_dis]
        
        phy_dis=line[786:1566]
        phy_dis=[float(x) for x in phy_dis]
        
        phy_dis_sorted=[y for x, y in sorted(zip(genetic_dis,phy_dis))]
        genetic_dis_sorted=[x for x, y in sorted(zip(genetic_dis,phy_dis))]
        feature=genetic_dis_sorted[:gen_len] + phy_dis_sorted[:phy_len]
        feature=np.asarray(feature)
        
        data.append(feature)
        parameter.append(params[2])


# In[11]:


c = list(zip(data,parameter))
shuffle(c)
data,parameter = zip(*c)


s=len(data)
s=int(0.9*s)



train_data=data[:s]
train_param=parameter[:s]

#test_data=data[s:]
#test_param=parameter[s:]

train_data=np.asarray(train_data)
train_param=np.asarray(train_param)


#test_data=np.asarray(test_data)
#test_param=np.asarray(test_param)


# In[12]:





inputs = Input(shape=((gen_len+phy_len),))

x = Dense(neurons_1,kernel_regularizer=regularizers.l2(0.00))(inputs)
x=LeakyReLU(alpha=0.1)(x)
x=Dropout(dropout_1)(x)
x = BatchNormalization()(x)
x = Dense(neurons_2, activation='relu',kernel_regularizer=regularizers.l2(0.00))(x)
x = Dense(neurons_2, activation='relu',kernel_regularizer=regularizers.l2(0.00))(x)
x=Dropout(dropout_2)(x)
x = Dense(neurons_3, activation='relu',kernel_regularizer=regularizers.l2(0.00))(x)


label_layer_1 = Input((1,))




output1 = Dense(1, activation='linear')(x)
output2=Dense(1,activation='softplus')(x)


var_layer=Lambda(lambda x: x + 1e-6)(output2)
var_log= K.log(var_layer)
var_log_half=Lambda(lambda x: x/2.0)(var_log)

subtracted = Subtract()([output1,label_layer_1])


divResult = Lambda(lambda x: x[0]/(x[1]*2))([K.square(subtracted),var_layer])

loss_var= var_log_half + divResult

vae_loss = K.mean(loss_var)

model = Model(input=[inputs,label_layer_1],output=[output1,output2])
model.add_loss(vae_loss)



optimizer1=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


model.compile(optimizer=optimizer1)

early_stopping_monitor = callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto',restore_best_weights=True)

time_callback = TimeHistory()

times = time_callback.times

#history=model.fit(train_data,[train_param0,train_param1,train_param2],batch_size=256,epochs=1000,validation_split=0.3,shuffle=True, callbacks=[early_stopping_monitor])
history=model.fit([train_data,train_param],batch_size=batch_size,epochs=2000,validation_split=0.20,shuffle=True,callbacks=[early_stopping_monitor,time_callback])
times = time_callback.times


score=model.evaluate([test_data,test_param])
print(score)

y_pred = model.predict([test_data,test_param])

print(test_data.shape)

pickle.dump(y_pred, open("y_pred_variance_model_radius"+".pkl.dat", "wb"))
#pickle.dump(y_pred[1], open("var_variance_model"+title+".pkl.dat", "wb"),protocol=2)

pickle.dump(test_param, open("test_param_variance_model_radius"+".pkl.dat", "wb"))

model_name1='saved_model_'+'.pickle.dat'



pickle.dump(model, open(model_name1, "wb"))


# In[19]:
'''

with open("y_pred_variance_model"+".pkl.dat", "rb") as f:
                test_data = pickle.load(f)
with open("test_param_variance_model"+".pkl.dat", "rb") as f:
                test_param = pickle.load(f)
        
print(test_data)
print(test_param)

'''
# In[ ]:





# In[ ]:




