from bigbird.bigbird.core import encoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib
from random import shuffle

from bigbird.bigbird.core import encoder


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

y = get_available_gpus()

subjects = [each for each in glob('./Data/'+"*")]
all_paths = [j+"/" for sub in [glob(x+'/'+"*") for x in subjects] for j in sub]
all_paths = all_paths[:2]
all_train_file_paths = []
all_test_file_paths = []

for each in all_paths:
    
    if each.split("/")[-2]=='train':
        
        for _ in glob(each+'*'):
            temp_class = _.split("/")[-1]                
            all_train_file_paths+=glob(_+"/*")
            
    if each.split("/")[-2]=='val':
        
        for _ in glob(each+'*'):
            temp_class = _.split("/")[-1]                
            all_test_file_paths+=glob(_+"/*")


shuffle(all_train_file_paths)
shuffle(all_test_file_paths)


classes = list(set([each.split("/")[-2] for each in all_train_file_paths]))

global window
window = 32
stride = int(window/2)
n_flicks = 40

temp = [np.load(x).shape[-1] for x in all_train_file_paths]

'''
print("Maximum and minimum signal length",max(temp),min(temp))

print(f"Max window size: {max(temp)/n_flicks} and small window size: {min(temp)/n_flicks} for 40 flicks in a file")

print(f"Average window size = {int(sum(temp)/len(temp)/40)}")

print("Number of train sample window slices", sum([int((1+(len(np.load(x).T)-window)/stride)) for x in all_train_file_paths]))
print("Number of test sample window slices", sum([int((1+(len(np.load(x).T)-window)/stride)) for x in all_test_file_paths]))
'''




def train_data_generator():
    file_paths = all_train_file_paths
    
    global window,stride

    classes = list(set([each.split("/")[-2] for each in file_paths]))
    for each in file_paths:

        sliced_data = []
        sliced_labels = []
        x = np.array(np.load(each).T)*10000
        #window = int(len(x)/n_flicks)
        
        # channel number starts with index !
        x = np.delete(x.T,3,0).T
        # actually the second bad channel is 31 but as we have already removed the channel the index will be 30
        x = np.delete(x.T,30,0).T
        start = 0
        end = start+window
        #stride = window*3

        #print(window,stride,start,end)
        label_onehot = np.zeros(20)
        label_onehot[classes.index(each.split("/")[-2])]=1
        y = np.array(label_onehot, dtype='float32')
        
        while end<len(x):
            data_slice = np.array(x[start:end])  
            sliced_data.append(np.array(data_slice, dtype='float32'))
            start = start + stride
            end = start + window

        shuffle(sliced_data)
        for X in sliced_data:
            yield ([X], [y])


def test_data_generator():
    file_paths = all_test_file_paths
    
    global window,stride
    
    classes = list(set([each.split("/")[-2] for each in file_paths]))
    for each in file_paths:

        sliced_data = []
        sliced_labels = []

        x = np.array(np.load(each).T)*10000
        #window = int(len(x)/n_flicks)

        # channel number starts with index !
        x = np.delete(x.T,3,0).T
        # actually the second bad channel is 31 but as we have already removed the channel the index will be 30
        x = np.delete(x.T,30,0).T

        start = 0
        end = start+window
        stride = window
        
        label_onehot = np.zeros(20)
        label_onehot[classes.index(each.split("/")[-2])]=1
        y = np.array(label_onehot, dtype='float32')
        
        while end<len(x):

            data_slice = np.array(x[start:end])

            sliced_data.append(np.array(data_slice, dtype='float32'))

            start = start + stride
            end = start + window
            
        shuffle(sliced_data)
        for X in sliced_data:
            yield ([X], [y])

input_shape = (window,120)

train_dataset = tf.data.Dataset.from_generator(
    generator=train_data_generator, 
    output_types=(np.float32, np.float32), 
    output_shapes=((None,input_shape[0],input_shape[1]),(None,20))
)


test_dataset = tf.data.Dataset.from_generator(
    generator=test_data_generator, 
    output_types=(np.float32, np.float32), 
    output_shapes=((None,input_shape[0],input_shape[1]),(None,20))
)

num_classes = len(classes)


x,y = train_data_generator(),test_data_generator()
st1 = sum(1 for _ in x)
st2 = sum(1 for _ in y)
del x,y



projection_dim = 120
n_transformer_layers = 4
num_heads = 10

Mask = None
drop_rate = 0.6

'''
  attention_type,
               hidden_size=768,
               intermediate_size=3072,
               intermediate_act_fn=utils.gelu,
               attention_probs_dropout_prob=0.0,
               hidden_dropout_prob=0.1,
               initializer_range=0.02,
               num_attention_heads=12,
               num_rand_blocks=3,
               seq_length=1024,
               block_size=64,
               use_bias=True,
               seed=None,
               name=None)
'''



inputs = layers.Input(shape=input_shape)

for i in range(n_transformer_layers):
    
  
  encoder_layer = encoder.PostnormEncoderLayer('original_full',
                                               hidden_size=projection_dim,
                                               intermediate_size=projection_dim,
                                               intermediate_act_fn=tf.keras.activations.relu,
                                               hidden_dropout_prob=drop_rate,
                                               seq_length=120,
                                               block_size=num_heads,
                                               num_attention_heads=num_heads,
                                               name='encd'+str(i))

  if i==0:
    emb = inputs
    encd = encoder_layer(emb)
    
  else:
    encd = encoder_layer(encd)




encd = layers.Flatten()(encd)


logits = layers.Dense(num_classes,activation='softmax')(encd)

model = keras.Model(inputs=inputs, outputs=logits)
model.summary()

learning_rate = 1e-4
batch = 40
n_epoch = 150




opt = Adam(learning_rate=learning_rate)
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('./bigbird_log.csv', append=True, separator=',')
#my_callbacks = tf.keras.callbacks.EarlyStopping(patience=10)

model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])

from datetime import datetime

now = datetime.now() # current date and time

date_time = now.strftime("%m:%d:%Y, %H:%M:%S")

# print(date_time)

# st1 = int(sum([int((1+(len(np.load(x).T)-window)/stride)) for x in all_train_file_paths]))
# st2 = int(sum([int((1+(len(np.load(x).T)-window)/stride)) for x in all_test_file_paths]))



hist = model.fit(train_dataset.repeat(),validation_data=test_dataset.repeat(),epochs=n_epoch,steps_per_epoch=st1,validation_steps=st2,shuffle=True,callbacks=[csv_logger])
'''

hist = model.fit(train_dataset,validation_data=test_dataset,epochs=n_epoch)#,shuffle=True,callbacks=[csv_logger,my_callbacks])
'''

model.save_weights('./BigBird')

fig,ax = plt.subplots(1,2,figsize=(15,5))

ax[0].title.set_text("Accuracy")
ax[0].plot(hist.history['accuracy'])

ax[1].title.set_text("Loss")
ax[1].plot(hist.history['loss'],c='C1')

# fig.savefig('./results_{0}_{1}_{2}_{3}.png'.format(date_time,projection_dim,n_transformer_layers,num_heads))
# plt.show()

# print(hist.history)

import pandas as pd

metrics = pd.read_csv('./bigbird_log.csv')
for x,y in zip(metrics.index[::-1],metrics['epoch'][::-1]):
    if y==0:
        break
metrics = metrics.iloc[x:]
metrics.reset_index(inplace=True)
metrics.drop('index',axis=1,inplace=True)

fig,ax = plt.subplots(1,2,figsize=(15,5))

ax[0].title.set_text("Accuracy")
ax[0].plot(metrics['accuracy'])
ax[0].plot(metrics['val_accuracy'])

ax[1].title.set_text("Loss")
ax[1].plot(metrics['loss'])
ax[1].plot(metrics['val_loss'])


fig.savefig('./results_{0}_{1}_{2}_{3}_{4}.png'.format(date_time,projection_dim,n_transformer_layers,num_heads,drop_rate))
plt.show()

