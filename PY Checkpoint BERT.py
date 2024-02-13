import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib
from random import shuffle


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

class MultiHeadAttention(tf.keras.layers.Layer):
    
  def scaled_dot_product_attention(self,q, k, v, mask):
      """Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead)
      but it must be broadcastable for addition.

      Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

      Returns:
        output, attention_weights
      """

      matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

      # scale matmul_qk
      dk = tf.cast(tf.shape(k)[-1], tf.float32)
      scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

      # add the mask to the scaled tensor.
      if mask is not None:
        scaled_attention_logits += (mask * -1e9)

      # softmax is normalized on the last axis (seq_len_k) so that the scores
      # add up to 1.
      attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

      output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

      return output, attention_weights

  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = self.scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights




class EncoderLayer(tf.keras.layers.Layer):
    
    
    def point_wise_feed_forward_network(self,d_model, dff):
        return tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
          tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
      ])

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=0.01,scale=False)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=0.01,scale=False)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        
        
        attn_output, _ = self.mha(x, x,x,None)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        f_out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        x = tf.reverse(x,axis=[1])
        attn_output, _ = self.mha(x, x,x,None)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        b_out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        b_out2 = tf.reverse(b_out2,axis=[1])
        out2 = tf.concat([f_out2,b_out2],axis=1)
        #out2 = tf.add(f_out2,b_out2)
        #out2 = layers.Dense(self.d_model)(merged_out2)
        
        return out2


x,y = train_data_generator(),test_data_generator()
st1 = sum(1 for _ in x)
st2 = sum(1 for _ in y)
del x,y

projection_dim = 120
n_transformer_layers = 4
num_heads = 10

Dropout_train = False
drop_rate = 0.4


inputs = layers.Input(shape=input_shape)


for i in range(n_transformer_layers):

    encoder_layer = EncoderLayer(d_model = projection_dim, num_heads=num_heads, dff=projection_dim,rate=drop_rate)

    if i ==0:
        encd = encoder_layer(inputs)
    else:
        encd = encoder_layer(encd)

#encd = layers.GlobalMaxPooling1D()(encd)
encd = layers.Flatten()(encd)


logits = layers.Dense(num_classes,activation='softmax')(encd)

model = keras.Model(inputs=inputs, outputs=logits)

model.summary()

learning_rate = 1e-3
batch = 50
n_epoch = 100




opt = Adam(learning_rate=learning_rate)
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('./BERT_log.csv', append=True, separator=',')
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

model.save_weights('./BERT')


fig,ax = plt.subplots(1,2,figsize=(15,5))

ax[0].title.set_text("Accuracy")
ax[0].plot(hist.history['accuracy'])

ax[1].title.set_text("Loss")
ax[1].plot(hist.history['loss'],c='C1')

# fig.savefig('./results_{0}_{1}_{2}_{3}.png'.format(date_time,projection_dim,n_transformer_layers,num_heads))
# plt.show()

# print(hist.history)

import pandas as pd

metrics = pd.read_csv('./BERT_log.csv')
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

