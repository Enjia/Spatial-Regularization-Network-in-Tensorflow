import tensorflow as tf

class Module(object):

    def __init__(self, inputs, is_train=True, trainable=True):
        self.inputs = inputs
        self.is_train = is_train
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError("Need to be implemented in subclass")

    @staticmethod
    def make_cpu_variable(name, shape, initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=True):
        with tf.device("/cpu:0"):
            var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
        return var

    def conv(self, x, k_h, k_w, c_o, s_h, s_w, name, relu, group=1, bias_term=False, padding='SAME', trainable=True):
        with tf.name_scope(name), tf.variable_scope(name):
            c_i = x.get_shape().as_list()[-1] / group
            weights = self.make_cpu_variable("weights", [k_h, k_w, c_i, c_o], trainable=trainable)
            def conv2d(i, w):
                return tf.nn.conv2d(i, w, [1, s_h, s_w, 1], padding)
            if group == 1:
                outputs = conv2d(x, weights)
            else:
                group_inputs = tf.split(x, num_or_size_splits=group, axis=3, name="split_inputs")
                group_weights = tf.split(weights, num_or_size_splits=group, axis=3, name="split_weights")
                group_outputs = [conv2d(i, w) for i, w in zip(group_inputs, group_weights)]
                outputs = tf.concat(group_outputs)
            if bias_term:
                biases = self.make_cpu_variable("biases", [c_o], trainable=trainable)
                outputs = tf.nn.bias_add(outputs, biases)
            if relu:
                outputs = tf.nn.relu(outputs)
            return outputs

    @staticmethod
    def max_pool(x, k_h, k_w, s_h, s_w, name, padding="VALID"):
        with tf.name_scope(name):
            outputs = tf.nn.max_pool(x, [1, k_h, k_w, 1], [1, s_h, s_w, 1], padding)
            return outputs

    @staticmethod
    def avg_pool(x, k_h, k_w, s_h, s_w, name, padding="VALID"):
        with tf.name_scope(name):
            outputs = tf.nn.avg_pool(x, [1, k_h, k_w, 1], [1, s_h, s_w, 1], padding)
            return outputs

    def fc(self, x, nout, name, relu, bias_term=False, trainable=True):
        with tf.name_scope(name), tf.variable_scope(name):
            input_shape = x.get_shape.as_list()
            if len(input_shape) == 4:
                dim = 1
                for d in input_shape[1:]:
                    dim *= d
                x = tf.reshape(x, dim)
            else:
                dim = input_shape[1]
            weights = self.make_cpu_variable("weights", [dim, nout],
                                             initializer=tf.truncated_normal_initializer(stddev=0.001),
                                             trainable=trainable)
            outputs = tf.matmul(x, weights)
            if bias_term:
                biases = self.make_cpu_variable("biases", [nout], trainable=trainable)
                outputs = tf.nn.bias_add(outputs, biases)
            if relu:
                outputs = tf.nn.relu(outputs)
            return outputs

    @staticmethod
    def drop(x, keep_prob, name):
        with tf.name_scope(name):
            outputs = tf.nn.dropout(x, keep_prob)
            return outputs

    @staticmethod
    def sigmoid(x, name):
        with tf.name_scope(name):
            outputs = tf.nn.sigmoid(x)
            return  outputs

    @staticmethod
    def softmax(x, name):
        with tf.name_scope(name):
            outputs = tf.nn.softmax(x)
            return outputs

    @staticmethod
    def batch_normal(x, is_train, name, activiation_fn):
        with tf.name_scope(name), tf.variable_scope(name):
            outputs = tf.contrib.layers.batch_norm(x, decay=0.999, scale=True,
                                                   activiation_fn=activiation_fn,
                                                   is_training=is_train)
            return outputs

    @staticmethod
    def element_wise_mul(x1, x2, name):
        with tf.name_scope(name):
            outputs = tf.multiply(x1, x2)
            return outputs

    @staticmethod
    def add(x, name):
        with tf.name_scope(name):
            outputs = tf.add_n(x)
            return outputs

    @staticmethod
    def relu(x, name):
        with tf.name_scope(name):
            outputs = tf.nn.relu(x)
            return outputs
