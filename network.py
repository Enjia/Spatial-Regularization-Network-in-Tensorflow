import tensorflow as tf
from module import Module

CLASSES = 81

class SRNet(Module):

    def setup(self):
        outputs = self._start_block()
        print("after start block:", outputs.shape)
        outputs = self._resblock(outputs, 256, "2a", identity_connection=False)
        outputs = self._resblock(outputs, 256, "2b")
        outputs = self._resblock(outputs, 256, "2c")
        print("after block1:", outputs.shape)
        outputs = self._resblock(outputs, 512, "3a", half_size=True, identity_connection=False)
        for i in range(3):
            outputs = self._resblock(outputs, 512, "3b%d" % i)
        print("after block2:", outputs.shape)
        outputs = self._resblock(outputs, 1024, "4a", half_size=True, identity_connection=False)
        for i in range(22):
            outputs = self._resblock(outputs, 1024, "4b%d" % i)
        print("after block3:", outputs.shape)
        outputs_res4b22_relu = outputs
        outputs = self._resblock(outputs, 2048, "5a", identity_connection=False)
        outputs = self._resblock(outputs, 2048, "5b")
        outputs = self._resblock(outputs, 2048, "5c")

        return outputs_res4b22_relu, outputs

    def main_net(self):
        _, outputs = self.setup()
        outputs = self.avg_pool(outputs, 7, 7, 1, 1, "avg_pool")
        ycls_raw_score = self.fc(outputs, CLASSES, "raw_score", relu=True)
        ycls_score = self.sigmoid(outputs, "score")
        return ycls_raw_score, ycls_score

    def srnet(self):
        outputs, _ = self.setup()
        outputs = self._resblock(outputs, 1024, "att1_base_branch1")
        outputs = self._resblock(outputs, 1024, "att1_base_branch2")
        outputs = self.batch_normal(outputs, is_train=self.is_train, name="att1_bn")
        outputs = self.relu(outputs, name="att1_relu")
        common_inputs = outputs

        #attention map blocks
        def sub_att_conv(x):
            output = self.conv(x, 1, 1, 512, 1, 1, name="att2_conv1")
            output = self.batch_normal(output, is_train=self.is_train, name="att2_bn1", activiation_fn=tf.nn.relu)
            output = self.conv(output, 3, 3, 512, 1, 1, name="att2_conv2")
            output = self.batch_normal(output, is_train=self.is_train, name="att2_bn2", activiation_fn=tf.nn.relu)
            output = self.conv(output, 1, 1, CLASSES, 1, 1, name="att2_conv3")
            output = tf.reshape(output, [-1, 1, 1, 196])
            output = tf.nn.softmax(output, axis=3)
            output = tf.reshape(output, [-1, 14, 14, CLASSES])
            return output

        def sub_conf_conv(x):
            output = self.conv(x, 1, 1, 512, 1, 1, name="conf2_conv1")
            output = self.batch_normal(output, is_train=self.is_train, name="conf2_bn1", activiation_fn=tf.nn.relu)
            output = self.conv(output, 1, 1, CLASSES, 1, 1, name="conf2_conv2")
            return output

        # calculate the score of y_sr
        def group_conv(x, out, k_h, k_w, group, name):
            group_size = out / group
            group_outputs = []
            for index in range(len(x[-1])):
                group_output = self.conv(x[:, :, :, index], k_h, k_w, group_size, 1, 1, name=name)
                group_outputs.append(group_output)
            group_outputs = tf.concat(group_outputs)
            return group_outputs

        #calculate the score of y_att
        att_map = sub_att_conv(common_inputs)
        conf_map = sub_conf_conv(common_inputs)

        #conf_map_sig is used for weighted attention maps
        conf_map_sig = self.sigmoid(conf_map)

        y_att = tf.multiply(att_map, conf_map, name="")
        y_att = tf.squeeze(tf.reshape(y_att, [-1, 1, 1, 196]))
        y_att = tf.reduce_sum(y_att, axis=1, keep_dims=True)
        y_att_raw_score = tf.reshape(y_att, [-1, CLASSES])
        y_att_score = self.sigmoid(y_att)

        y_sr = tf.multiply(att_map, conf_map_sig)
        y_sr = self.conv(y_sr, 1, 1, 512, 1, 1, name="comb_conv1")
        y_sr = self.batch_normal(y_sr, is_train=self.is_train, name="comb_bn1", activiation_fn=tf.nn.relu)
        y_sr = self.conv(y_sr, 1, 1, 512, 1, 1, name="comb_conv2")
        y_sr = self.batch_normal(y_sr, is_train=self.is_train, name="comb_bn2", activiation_fn=tf.nn.relu)
        y_sr = group_conv(y_sr, 2048, 512, 14, 14, name="comb_conv3")
        y_sr = self.batch_normal(y_sr, is_train=self.is_train, name="comb_bn3", activiation_fn=tf.nn.relu)
        y_sr_raw_score = self.fc(y_sr, CLASSES, "comb_fc4", relu=False)

        return y_att_raw_score, y_att_score, y_sr_raw_score

    def _start_block(self, x, name):
        with tf.name_scope(name), tf.variable_scope(name):
            outputs = self.conv(x, 7, 7, 64, 2, 2, "conv1", relu=False)
            outputs = self.batch_normal(outputs, name="batch_conv1",
                                        is_train=self.,
                                        activiation_fn=tf.nn.relu)
            outputs = self.max_pool(outputs, 3, 3, 3, 3, name="pool1")
            return outputs

    def _resblock(self, x, c_o, name, half_size=False, identity_connection=True):
        s = 2 if half_size else 1
        assert c_o % 4 == 0, "Bottleneck number of output ERROR!"
        #branch1
        if not identity_connection:
            o_b1 = self.conv(x, 1, 1, c_o, s, s, name="res%s_branch1" % name)
            o_b1 = self.batch_normal(o_b1, is_train=self.is_train, name="bn%s_branch1" % name, activiation_fn=None)
        else:
            o_b1 = x
        #branch2
        o_b2a = self.conv(o_b1, 1, 1, c_o / 4, s, s, name="res%s_branch2a" % name)
        o_b2a = self.batch_normal(o_b2a, is_train=self.is_train, name="bn%s_branch2a" % name, activiation_fn=tf.nn.relu)

        o_b2b = self.conv(o_b2a, 1, 1, c_o / 4, s, s, name="res%s_branch2b" % name)
        o_b2b = self.batch_normal(o_b2b, is_train=self.is_train, name="bn%s_branch2b" % name, activiation_fn=tf.nn.relu)

        o_b2c = self.conv(o_b2b, 1, 1, c_o, s, s, name="res%s_branch2c" % name)
        o_b2c = self.batch_normal(o_b2c, is_train=self.is_train, name="bn%s_branch2c" % name, activiation_fn=None)
        #add
        outputs = self.add([o_b1, o_b2c], name="res%s" % name)
        outputs = self.relu(outputs, name="res&s_relu" % name)

        return outputs


