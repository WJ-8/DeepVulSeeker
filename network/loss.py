import random

import numpy as np
import tensorflow as tf
from keras import backend as K

# 分类问题的类数，fc层的输出单元个数
NUM_CLASSES = 2
# 更新中心的学习率
ALPHA = 0.6
# center-loss的系数
LAMBDA = 0.0005

tf.compat.v1.disable_v2_behavior()


def center_loss(labels, features, alpha=ALPHA, num_classes=NUM_CLASSES):
    """
    获取center loss及更新样本的center
    :param labels: Tensor,表征样本label,非one-hot编码,shape应为(batch_size,).
    :param features: Tensor,表征样本特征,最后一个fc层的输出,shape应该为(batch_size, num_classes).
    :param alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
    :param num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    :return: Tensor, center-loss， shape因为(batch_size,)
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.compat.v1.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                                        initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，如果labels已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])
    labels = K.cast(labels, 'int32')
    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    # 更新centers
    centers_update_op = tf.compat.v1.scatter_sub(centers, labels, diff)

    # 这里使用tf.control_dependencies更新centers
    with tf.control_dependencies([centers_update_op]):
        # 计算center-loss
        c_loss = tf.nn.l2_loss(features - centers_batch)

    return c_loss


def softmax_loss(labels, features):
    """
    计算softmax-loss
    :param labels: 等同于y_true，使用了one_hot编码，shape应为(batch_size, NUM_CLASSES)
    :param features: 等同于y_pred，模型的最后一个FC层(不是softmax层)的输出，shape应为(batch_size, NUM_CLASSES)
    :return: 多云分类的softmax-loss损失，shape为(batch_size, )
    """
    return K.sparse_categorical_crossentropy(labels, K.softmax(features, axis=-1))


def loss(labels, features):
    """
    使用这个函数来作为损失函数，计算softmax-loss加上一定比例的center-loss
    :param labels: Tensor，等同于y_true，使用了one_hot编码，shape应为(batch_size, NUM_CLASSES)
    :param features: Tensor， 等同于y_pred, 模型的最后一个fc层(不是softmax层)的输出，shape应为(batch_size, NUM_CLASSES)
    :return: softmax-loss加上一定比例的center-loss
    """
    labels = K.cast(labels, dtype=tf.float32)
    # 计算softmax-loss
    sf_loss = softmax_loss(labels, features)
    # 计算center-loss，因为labels使用了one_hot来编码，所以这里要使用argmax还原到原来的标签
    c_loss = center_loss(labels, features)
    return sf_loss + LAMBDA * c_loss


def categorical_accuracy(y_true, y_pred):
    """
    重写categorical_accuracy函数，以适应去掉softmax层的模型
    :param y_true: 等同于labels，
    :param y_pred: 等同于features。
    :return: 准确率
    """
    # 计算y_pred的softmax值
    sm_y_pred = K.softmax(y_pred, axis=-1)
    y_true = K.cast(y_true, 'int64')
    # 返回准确率
    return K.cast(K.equal(y_true, K.argmax(sm_y_pred, axis=-1)), K.floatx())
