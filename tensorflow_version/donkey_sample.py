import tensorflow as tf
import sys
sys.path.append('.')
from donkey import Donkey
import matplotlib.pyplot as plt
import numpy as np

image_batch, length_batch, digits_batch = Donkey.build_batch(r'E:\DataSets\SVHN\train.tfrecords',
                                                             num_example=100,
                                                             batch_size=36,
                                                             shuffled=False)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    image_batch_val, length_batch_val, digits_batch_val = sess.run([image_batch, length_batch, digits_batch])
    image_batch_val = (image_batch_val / 2.0) + 0.5
    print(np.array(image_batch_val[0]).shape)
    fig, axes = plt.subplots(6, 6, figsize=(20, 20))
    for i, ax in enumerate(axes.flat):
        title = 'length: %d\ndigits= %d, %d, %d, %d, %d' % (length_batch_val[i],
                                                            digits_batch_val[i][0],
                                                            digits_batch_val[i][1],
                                                            digits_batch_val[i][2],
                                                            digits_batch_val[i][3],
                                                            digits_batch_val[i][4])
        ax.imshow(image_batch_val[i])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    coord.request_stop()
    coord.join(threads)
