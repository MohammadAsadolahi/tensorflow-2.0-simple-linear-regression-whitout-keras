import tensorflow as tf
import numpy as np
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=x*3+7
learning_rate=0.001

X=tf.constant(tf.cast(x,dtype=tf.float32))
X=tf.reshape(X,shape=(1,10))

weights=tf.Variable(tf.random.uniform([10,1],minval=-1,maxval=1),dtype=tf.float32)#
bias=tf.Variable(tf.random.uniform([1]))

def linearRegressor(Xvector):
  result=tf.add(tf.reduce_sum(tf.matmul(weights,Xvector),axis=0),bias)
  return result

def loss(Y_pred,Y_target):
  return tf.reduce_mean(tf.square(Y_pred-Y_target))

for epoch in range(4000):
  with tf.GradientTape() as tape:
    result=linearRegressor(X)
    cost=loss(result,y)
    gradients = tape.gradient(cost, [weights,bias])
    weights.assign_sub(gradients[0]*learning_rate)
    bias.assign_sub(gradients[1]*learning_rate)
    print(f"epoch:{epoch}  loss:",cost.numpy())

print(weights,bias)

print(linearRegressor([[10,20]]))

