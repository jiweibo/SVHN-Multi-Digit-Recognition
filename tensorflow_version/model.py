import tensorflow as tf


class Model(object):

    @staticmethod
    def inference(x, drop_rate):
        with tf.variable_scope('hidden1'):
            conv = tf.layers.conv2d(x, filters=48, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            hidden1 = pool  # 27 * 27 * 48
            # print(hidden1.shape)

        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            hidden2 = pool  # 27 * 27 * 64
            # print(hidden2.shape)

        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            hidden3 = pool  # 14 * 14 * 128
            # print(hidden3.shape)

        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(hidden3, filters=160, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            hidden4 = pool  # 14 * 14 *160
            # print(hidden4.shape)

        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            hidden5 = pool  # 7 * 7 * 192
            # print(hidden5.shape)

        with tf.variable_scope('hidden6'):
            conv = tf.layers.conv2d(hidden5, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            hidden6 = pool  # 7 * 7 * 192
            # print(hidden6.shape)

        with tf.variable_scope('hidden7'):
            conv = tf.layers.conv2d(hidden6, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            hidden7 = pool  # 4 * 4 * 192
            # print(hidden7.shape)

        with tf.variable_scope('hidden8'):
            conv = tf.layers.conv2d(hidden7, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            hidden8 = pool  # 4 * 4 * 192

        flatten = tf.reshape(hidden8, [-1, 4 * 4 * 192])

        with tf.variable_scope('hidden9'):
            dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.relu)
            dropout = tf.layers.dropout(dense, rate=drop_rate)
            hidden9 = dropout

        with tf.variable_scope('hidden10'):
            dense = tf.layers.dense(hidden9, units=3072, activation=tf.nn.relu)
            dropout = tf.layers.dropout(dense, rate=drop_rate)
            hidden10 = dropout

        with tf.variable_scope('digit_length'):
            dense = tf.layers.dense(hidden10, units=7)
            length = dense

        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(hidden10, units=11)
            digit1 = dense

        with tf.variable_scope('digit2'):
            dense = tf.layers.dense(hidden10, units=11)
            digit2 = dense

        with tf.variable_scope('digit3'):
            dense = tf.layers.dense(hidden10, units=11)
            digit3 = dense

        with tf.variable_scope('digit4'):
            dense = tf.layers.dense(hidden10, units=11)
            digit4 = dense

        with tf.variable_scope('digit5'):
            dense = tf.layers.dense(hidden10, units=11)
            digit5 = dense

        length_logist, digits_logist = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
        return length_logist, digits_logist

    @staticmethod
    def loss(length_logits, digits_logits, length_labels, digits_labels):
        # length_labels = tf.one_hot(length_labels, 7)
        # print(length_logits.shape, length_labels.shape)
        length_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
        return loss
