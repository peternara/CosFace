import tensorflow as tf
import numpy as np

###loss的py，因为你要用cosloss，所以loss函数要自己去定义，这就是自己定义的loss函数

def py_func(func, inp, Tout, stateful = True, name=None, grad_func=None):
    rand_name = 'PyFuncGrad' + str(np.random.randint(0,1E+8))
    tf.RegisterGradient(rand_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({'PyFunc':rand_name}):       #自定义反向传播中的梯度值
        return tf.py_func(func,inp,Tout,stateful=stateful, name=name)#该函数重构一个python函数，并将其作为一个TensorFlow的op使用。给定一个输入和输出都是numpy数组的python函数’func’，py_func函数将func重构进TensorFlow的计算图中。


def coco_forward(xw, y, m, name=None):# 所有的矩阵给标签所在的那一列减m
    # #pdb.set_trace()    比如（就把第三列）：[[2.   3.   7.75]
    #                                        [5.   4.   8.75]
    #                                        [7.   6.   9.75]]
    xw_copy = xw.copy()
    num = len(y)
    orig_ind = range(num)
    xw_copy[orig_ind,y] -= m
    return xw_copy

def coco_help(grad,y):
    grad_copy = grad.copy()
    return grad_copy

def coco_backward(op, grad):  

    y = op.inputs[1]
    m = op.inputs[2]
    grad_copy = tf.py_func(coco_help,[grad,y],tf.float32)
    return grad_copy,y,m

def coco_func(xw,y,m, name=None):
    with tf.op_scope([xw,y,m],name,"Coco_func") as name:
        coco_out = py_func(coco_forward,[xw,y,m],tf.float32,name=name,grad_func=coco_backward)
        return coco_out

def cos_loss(x, y,  num_cls, reuse=False, alpha=0.25, scale=64,name = 'cos_loss'):
    '''
    x: B x D - features
    y: B x 1 - labels
    num_cls: 1 - total class number
    alpah: 1 - margin
    scale: 1 - scaling paramter
    '''
    # define the classifier weights
    xs = x.get_shape()  #取x有多少列
    with tf.variable_scope('centers_var',reuse=reuse) as center_scope:
        w = tf.get_variable("centers", [xs[1], num_cls], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),trainable=True)

    #normalize the feature and weight
    #(N,D)
    x_feat_norm = tf.nn.l2_normalize(x,1,1e-10)
    #(D,C)
    w_feat_norm = tf.nn.l2_normalize(w,0,1e-10)

    # get the scores after normalization
    #(N,C)
    xw_norm = tf.matmul(x_feat_norm, w_feat_norm)
    #implemented by py_func
    #value = tf.identity(xw)
    #substract the marigin and scale it
    value = coco_func(xw_norm,y,alpha) * scale

    #implemented by tf api
    #margin_xw_norm = xw_norm - alpha
    #label_onehot = tf.one_hot(y,num_cls)
    #value = scale*tf.where(tf.equal(label_onehot,1), margin_xw_norm, xw_norm)


    # compute the loss as softmax loss
    cos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=value))

    return cos_loss
