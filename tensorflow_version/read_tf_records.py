import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

filename = r'E:\DataSets\SVHN\test.tfrecords'
filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'image': tf.FixedLenFeature([], tf.string),
                                       'length': tf.FixedLenFeature([], tf.int64),
                                       'digits': tf.FixedLenFeature([5], tf.int64)
                                   })
image = tf.decode_raw(features['image'], tf.uint8)
image = tf.reshape(image, [64, 64, 3])
length = features['length']
digits = features['digits']

sess = tf.InteractiveSession()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

(image_val, length_val, digits_val) = sess.run([image, length, digits])
print('length: %d, digits: %d, %d, %d, %d, %d' % (
    length_val, digits_val[0], digits_val[1], digits_val[2], digits_val[3], digits_val[4]
))
print(image_val)
print(np.array(image_val).shape)
plt.imshow(image_val)
plt.show()
coord.request_stop()
coord.join(threads)
sess.close()
