import tensorflow as tf
from model import Model
import matplotlib.pyplot as plt
from PIL import Image

tf.app.flags.DEFINE_string('image', None, 'Path to image file')
tf.app.flags.DEFINE_string('restore_checkpoints', None,
                           'Path to restore checkpoint (with out postfix), e.g. ./logs/train/model.ckpt-100')
FLAGS = tf.app.flags.FLAGS


def main(_):
    path_to_image_file = FLAGS.image
    path_to_restore_checkpoints = FLAGS.restore_checkpoints

    image = tf.image.decode_jpeg(tf.read_file(path_to_image_file), channels=3)
    image = tf.image.resize_images(image, [64, 64])
    image = tf.reshape(image, [64, 64, 3])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.multiply(tf.subtract(image, 0.5), 2)
    image = tf.image.resize_images(image, [54, 54])
    images = tf.reshape(image, [1, 54, 54, 3])

    length_logits, digits_logits = Model.inference(images, drop_rate=0.0)
    length_predictions = tf.argmax(length_logits, axis=1)
    digits_predictions = tf.argmax(digits_logits, axis=2)
    digits_predictions_string = tf.reduce_join(tf.as_string(digits_predictions), axis=1)

    with tf.Session() as sess:
        restorer = tf.train.Saver()
        restorer.restore(sess, path_to_restore_checkpoints)

        length_predictions_val, digits_predictions_string_val, digits_predictions_val = sess.run(
            [length_predictions, digits_predictions_string, digits_predictions])
        title = 'length: %d\ndigits= %d, %d, %d, %d, %d' % (length_predictions_val[0],
                                                            digits_predictions_val[0][0],
                                                            digits_predictions_val[0][1],
                                                            digits_predictions_val[0][2],
                                                            digits_predictions_val[0][3],
                                                            digits_predictions_val[0][4])
        img = Image.open(path_to_image_file, 'r')
        plt.imshow(img)
        plt.title(title)
        plt.show()
        # print('length: %d' % length_predictions_val[0])
        # print('digits: %s', digits_predictions_string_val[0])


if __name__ == '__main__':
    tf.app.run(main=main)
