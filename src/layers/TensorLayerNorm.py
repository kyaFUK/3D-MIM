#import tensorflow as tf
import tensorflow.compat.v1 as tf  
EPSILON = 0.00001

#added the case of dims=6
def tensor_layer_norm(x, state_name):
    x_shape = x.get_shape()
    dims = x_shape.ndims
    params_shape = x_shape[-1:]
    if dims == 4:
        m, v = tf.nn.moments(x, [1,2,3], keep_dims=True)
    elif dims == 5:
        m, v = tf.nn.moments(x, [1,2,3,4], keep_dims=True)
    elif dims == 6:
        m, v = tf.nn.moments(x, [1,2,3,4,5], keep_dims=True)
    elif dims == 2:
        m, v = tf.nn.moments(x, [1], keep_dims=True)
    else:
        raise ValueError('input tensor for layer normalization must be rank 4, 5, or 6.')
    b = tf.get_variable(state_name+'b',initializer=tf.zeros(params_shape))
    s = tf.get_variable(state_name+'s',initializer=tf.ones(params_shape))
    x_tln = tf.nn.batch_normalization(x, m, v, b, s, EPSILON)
    return x_tln
