#-*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
#下载并载入mnst手写数据库
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('mnist_data',one_hot=True)

#None表示张量（Tensor）的地一个维度可以是任何长度
input_x=tf.placeholder(tf.float32,[None,28*28])/255
output_y=tf.placeholder(tf.int32,[None,10])
input_x_images=tf.reshape(input_x,[-1,28,28,1])

#从Test 数据集中选取3000个手写数字的图片和对应标签
test_x=mnist.test.images[:3000]
test_y=mnist.test.labels[:3000]


#构建神经网络
#构第一层卷积
conv1=tf.layers.conv2d(
      inputs=input_x_images,
      filters=32,
      kernel_size=[5,5],
      strides=1,
      padding='same',
      activation=tf.nn.relu
)

#第1层池化
pool1=tf.layers.max_pooling2d(
      inputs=conv1,
      pool_size=[2,2],
      strides=2,
      padding='same'
)

#构建第2层卷积
conv2=tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5,5],
      strides=1,
      padding='same',
      activation=tf.nn.relu
)

#第2层池化
pool2=tf.layers.max_pooling2d(
      inputs=conv2,
      pool_size=[2,2],
      strides=2,
      padding='same'
)
#平坦化

flat=tf.reshape(pool2,[-1,7*7*64])

#全连接层

dense=tf.layers.dense(inputs=flat,units=1024,activation=tf.nn.relu)

#dropout,rate=0.5,
dropout=tf.layers.dropout(inputs=dense,rate=0.5)

#10个神经元的全连接层，这里不用激活函数进行非线性化
logits=tf.layers.dense(inputs=dropout,units=10)

#计算误差（计算交叉熵，再用softmax计算百分比概率）
loss=tf.losses.softmax_cross_entropy(onehot_labels=output_y,logits=logits)

#Adam优化器，学习率0.001
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#精度。计算 预测值 与 实际标签 的匹配程度
#返回(accuracy, update_op),会创建两个  局部变量
accuracy=tf.metrics.accuracy(labels=tf.argmax(output_y,axis=1),predictions=tf.argmax(logits,axis=1))[1]

#创建会话
sess=tf.Session()
#初始化变量：
init=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init)

for i in range(20000):
    batch=mnist.train.next_batch(50)#
    train_loss,train_op_=sess.run([loss,train_op],{input_x:batch[0],output_y:batch[1]})
    if i%100==0:
        test_accuracy=sess.run(accuracy,{input_x:test_x,output_y:test_y})
        print("step=%d,Train loss=%.4f,[Test accuracy=%.2f]")%(i,train_loss,test_accuracy)

#测试：打印 20个 预测值 和 真实值 的对
test_output=sess.run(logits,{input_x:test_x[:20]})
inferenced_y=np.argmax(test_output,1)
print(inferenced_y,'inferenced numbers')#推测的数字
print(np.argmax(test_y[:20],1),'real numbers')#真实的数字

