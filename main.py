import tensorflow as tf
import tensorflow.contrib.slim as slim
import myloss
import read_data

batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'scale':True,
        'is_training': True,
        'updates_collections': None,
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
} #如果你要用batch norml 这就是参数。最好要使用，我开始不适用loss无法收敛。

def prelu(x):
    with tf.variable_scope('PRelu'):
        alphas = tf.get_variable(name='prelu_alphas',initializer=tf.constant(0.25,dtype=tf.float32,shape=[x.get_shape()[-1]]))
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5
        return pos + neg

def res_part(res_input,output_channels):
    res_1=slim.conv2d(res_input,output_channels,kernel_size=3,stride=1,activation_fn=prelu,normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params)
    res_1=slim.conv2d(res_1,output_channels,kernel_size=3,stride=1, activation_fn=prelu,normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params)
    return res_1+res_input

#这段被注释的代码是我刚开始测试的代码，我想看看如果就输入一张图片，整个网络可以运行吗。
# image_raw_data = tf.read_file('F:/picture/ha.jpg')
#
# image = tf.image.decode_jpeg(image_raw_data,channels=3) #图片解码
# image = tf.reshape(image, (1, 112, 112, 3))
# image = tf.image.convert_image_dtype(image, dtype=tf.float32)

image = tf.placeholder(tf.float32,[64,112,112,3])
y = tf.placeholder(tf.int32,[64])

with tf.variable_scope('conv1'):
     myinput=slim.conv2d(image,64,kernel_size=3,stride=2,activation_fn=prelu,normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params)
     myinput=res_part(myinput,64)

with tf.variable_scope('conv2'):
     myinput=slim.conv2d(myinput,128,kernel_size=3,stride=2,activation_fn=prelu,normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params)
     myinput=res_part(myinput,128)
     myinput=res_part(myinput,128)

with tf.variable_scope('conv3'):
     myinput=slim.conv2d(myinput,256,kernel_size=3,stride=2,activation_fn=prelu,normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params)
     myinput=res_part(myinput,256)
     myinput=res_part(myinput,256)
     myinput=res_part(myinput,256)
     myinput=res_part(myinput,256)

with tf.variable_scope('conv4'):
     myinput=slim.conv2d(myinput,512,kernel_size=3,stride=2,activation_fn=prelu,normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params)
     myinput=res_part(myinput,512)


with tf.variable_scope('last'):
    myinput=slim.flatten(myinput)
    myinput=slim.fully_connected(myinput,512,activation_fn=None,normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params)



loss=myloss.cos_loss(myinput,y,2001)# 标签是y，一共有2001个类
#loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=myinput))

#学习率
global_step = tf.Variable(0)

learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)

######

optimizer=tf.train.GradientDescentOptimizer(learning_rate)

train=optimizer.minimize(loss)

#grads_and_vars = optimizer.compute_gradients(loss) #计算所有的梯度

init = tf.global_variables_initializer()



image_batch, label_batch =  read_data.read_and_decode("F:\\pythondata\\train.tfrecords",64)

#init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
thread = tf.train.start_queue_runners(sess=sess)
for i in range(10000):
    #print(sess.run(learning_rate,feed_dict={global_step:i}))
    images, labels = sess.run([image_batch, label_batch])
    sess.run(train,feed_dict={image:images,y:labels,global_step:i})

    if i%100==0:
        print(sess.run(loss,feed_dict={image:images,y:labels}))
        print(sess.run(learning_rate,feed_dict={global_step:i}))





