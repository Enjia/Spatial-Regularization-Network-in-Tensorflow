import tensorflow as tf

CROP_SIZE = (224, 224)
IMG_CHANNEL = 'BGR'

def image_mirror(img):
    distort_left_right_random = tf.random_uniform([], 0, 1.0, dtype=tf.float32)
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    return img

def image_crop(img):
    crop_h = CROP_SIZE[0]
    crop_w = CROP_SIZE[1]
    img = tf.image.resize_images(img, [crop_h, crop_w])
    return img

def input_stream(record_path, is_training, scope=None)
    with tf.device("/cpu:0"):
        with tf.variable_scope(scope or "input_stream"):
            reader = tf.TFRecordReader()
            filename_queue = tf.train.string_input_producer([record_path], None)
            _, record_value = reader.read(filename_queue)
            features = tf.parse_single_example(record_value,
                {
                    'image_jpeg': tf.FixedLenFeature([], tf.string),
                    'black': tf.FixedLenFeature([], tf.int64),
                    'blue': tf.FixedLenFeature([], tf.int64),
                    'dress': tf.FixedLenFeature([], tf.int64),
                    'jeans': tf.FixedLenFeature([], tf.int64),
                    'red': tf.FixedLenFeature([], tf.int64),
                    'shirt': tf.FixedLenFeature([], tf.int64)
                })
            img = tf.cast(tf.image.decode_jpeg(features['image_jpeg'], channels=3), tf.float32)
            label = [features['black'], features['blue'], features['dress'],
                     features['jeans'], features['red'],features['shirt']]
            label = tf.cast(label, tf.float32)
            if IMG_CHANNEL == 'BGR':
                img_r, img_g, img_b = tf.split(img, num_or_size_splits=3, axis=2)
                img = tf.cast(tf.concat([img_b, img_g, img_r], axis=2), dtype=tf.float32)
            if is_training:
                img = image_crop(img)
                img -= 127
                img = image_mirror(img)
            img.set_shape([CROP_SIZE[0], CROP_SIZE[1], 3])
            label.set_shape([6])
            stream = {'image':img, 'label': label}
            return stream

class Reader(object):

    def __init__(self, record_path, is_training=True):
        self.record_path = record_path
        self.stream = input_stream(self.record_path, is_training=is_training)

    def dequeue(self, batch_size):
        train_batch = tf.train.batch(self.stream, batch_size=batch_size)
        return train_batch
