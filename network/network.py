from keras import initializers
from keras import activations
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import keras

def convolutional2D(x,num_filters,kernel_size,resampling,strides=2):
    if resampling is 'up':
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=1, padding='same',
                       kernel_initializer=tf.keras.initializers.RandomNormal())(x)
        #x = keras.layers.Conv2DTranspose(num_filters,kernel_size=kernel_size, strides=2,  padding='same',
        #              kernel_initializer=keras.initializers.RandomNormal())(x)
    elif resampling is 'down':
        x = keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides,  padding='same',
                       kernel_initializer=tf.keras.initializers.RandomNormal())(x)
    return x


def ResBlock(x, num_filters, resampling, strides=2):
    # F1,F2,F3 = num_filters
    X_shortcut = x

    # //up or down
    x = convolutional2D(x, num_filters, kernel_size=(3, 3), resampling=resampling, strides=strides)

    # //BN_relu
    x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Activation('relu')(x)
    x = keras.layers.LeakyReLU()(x)

    # //cov2d
    x = keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=1, padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal())(x)

    # //BN_relu
    x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Activation('relu')(x)
    x = keras.layers.LeakyReLU()(x)

    # //cov2d
    x = keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=1, padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal())(x)
    # //BN
    x = keras.layers.BatchNormalization()(x)

    # //add_shortcut
    X_shortcut = convolutional2D(X_shortcut, num_filters, kernel_size=(1, 1), resampling=resampling, strides=strides)
    X_shortcut = keras.layers.BatchNormalization()(X_shortcut)

    X_add = keras.layers.Add()([x, X_shortcut])
    # X_add = keras.layers.Activation('relu')(X_add)
    X_add = keras.layers.LeakyReLU()(X_add)

    return X_add


class MyMultiHeadAttention(Layer):
    def __init__(self, output_dim, num_head, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.num_head = num_head
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(MyMultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(self.num_head, 3, input_shape[2], self.output_dim),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.Wo = self.add_weight(name='Wo',
                                  shape=(self.num_head * self.output_dim, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        self.built = True

    def call(self, x):
        q = K.dot(x, self.W[0, 0])
        k = K.dot(x, self.W[0, 1])
        v = K.dot(x, self.W[0, 2])
        e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘 [0, 2, 1]宽度和高度交换
        e = e / (self.output_dim ** 0.5)
        e = K.softmax(e)
        outputs = K.batch_dot(e, v)
        for i in range(1, self.W.shape[0]):
            q = K.dot(x, self.W[i, 0])
            k = K.dot(x, self.W[i, 1])
            v = K.dot(x, self.W[i, 2])
            # print('q_shape:'+str(q.shape))
            e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘
            e = e / (self.output_dim ** 0.5)
            e = K.softmax(e)
            # print('e_shape:'+str(e.shape))
            o = K.batch_dot(e, v)
            outputs = K.concatenate([outputs, o])
        z = K.dot(outputs, self.Wo)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "output_dim": self.output_dim,
            'num_head': self.num_head,
            'kernel_initializer': self.kernel_initializer
        })
        return config
