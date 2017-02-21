
# coding: utf-8

# In[1]:

import numpy as np
csv = np.genfromtxt ('train.csv', delimiter=",")


# In[2]:

mnist_train_labels_temp = csv[1:,0]
mnist_train_labels_temp.shape[0]


# In[3]:

m = mnist_train_labels_temp.shape[0]


# In[4]:

mnist_train_labels = np.zeros((m,11))
mnist_train_labels.shape


# In[5]:

mnist_train_labels[:,0] = mnist_train_labels_temp


# In[6]:

for x in range(m):
    i = mnist_train_labels[x,0]
    mnist_train_labels[x,i+1] = 1


# In[7]:

mnist_train_labels = mnist_train_labels[:,1:]
mnist_train_labels = mnist_train_labels.astype(int)
mnist_train_labels


# In[8]:

mnist_train_data = csv[1:,1:]
mnist_train_data = mnist_train_data.astype(int)
mnist_train_data


# In[9]:

import tensorflow as tf
sess = tf.InteractiveSession()


# In[10]:

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# In[11]:

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# In[12]:

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[13]:

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
#


# In[14]:

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# In[15]:

x_image = tf.reshape(x, [-1,28,28,1])


# In[16]:

phase_train = tf.placeholder(tf.bool, name='phase_train')
conv1 = conv2d(x_image, W_conv1) + b_conv1
conv1_bn = batch_norm(conv1, 32, phase_train)
h_conv1 = tf.nn.relu(conv1_bn)
h_pool1 = max_pool_2x2(h_conv1)


# In[17]:

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])


conv2 = conv2d(h_pool1, W_conv2) + b_conv2
conv2_bn = batch_norm(conv2, 64,  phase_train)
h_conv2 = tf.nn.relu(conv2_bn)
h_pool2 = max_pool_2x2(h_conv2)


# In[18]:

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[19]:

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[20]:

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# In[21]:

csv = np.genfromtxt('test.csv', delimiter=",")


# In[22]:

mnist_test_data = csv[1:,:]
mnist_test_data = mnist_test_data.astype(int)
mnist_test_data


# In[23]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer(), feed_dict={phase_train:True})
val = 1
for i in range(20480):
  batch_image = mnist_train_data[(val-1)*50:val*50-1,:]
  batch_label = mnist_train_labels[(val-1)*50:val*50-1,:]  
   #batch = mnist.train.next_batch(50) 
  val=val+1;  
  if val == m/50:
    val = 1
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch_image, y_: batch_label, keep_prob: 1.0,  phase_train: False})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch_image, y_:batch_label, keep_prob: 0.5, phase_train: True})


# In[24]:

classification = sess.run(tf.argmax(y_conv, 1), feed_dict={x: mnist_test_data, keep_prob: 1.0, phase_train:False})


# In[25]:

classification


# In[26]:

m_test = mnist_test_data.shape[0]
m_test


# In[27]:

mnist_test_result = np.zeros((m_test,2))
mnist_test_result.shape


# In[28]:

for j in range(m_test):
    mnist_test_result[j,0] = j+1


# In[29]:

mnist_test_result[:,1] = classification


# In[30]:

mnist_test_result[:10,:]


# In[31]:

import pandas as pd
names = ['imageid','label']
df = pd.DataFrame(mnist_test_result, columns=names)
df['imageid'] = df['imageid'].astype(int);
df['label'] = df['label'].astype(int);
df.head(10)


# In[32]:

df.to_csv('bde_assignment1_result_v6.csv', header=True, index=False ,sep=',')


# In[ ]:



