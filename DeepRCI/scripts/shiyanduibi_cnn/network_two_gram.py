import tensorflow as tf


class CMM_NET(tf.keras.Model):

    def __init__(self):
        super(CMM_NET, self).__init__()

        self.conv_1 = tf.keras.layers.Conv1D(16, kernel_size=2, padding='same', activation=tf.nn.relu)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_1_1 = tf.keras.layers.Conv1D(16, kernel_size=2, padding='same', activation=tf.nn.relu)
        self.bn_1_1 = tf.keras.layers.BatchNormalization()
        self.maxpool_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')


        self.conv_2 = tf.keras.layers.Conv1D(32, kernel_size=2, padding='same', activation=tf.nn.relu)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.conv_2_2 = tf.keras.layers.Conv1D(32, kernel_size=2, padding='same', activation=tf.nn.relu)
        self.bn_2_2 = tf.keras.layers.BatchNormalization()
        self.maxpool_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')


        self.conv_3 = tf.keras.layers.Conv1D(64, kernel_size=2, padding='same', activation=tf.nn.relu)
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.conv_3_3 = tf.keras.layers.Conv1D(64, kernel_size=2, padding='same', activation=tf.nn.relu)
        self.bn_3_3 = tf.keras.layers.BatchNormalization()
        self.maxpool_3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')


        self.conv_4 = tf.keras.layers.Conv1D(80, kernel_size=2, padding='same', activation=tf.nn.relu)
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.maxpool_4 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')

        self.conv_5 = tf.keras.layers.Conv1D(80, kernel_size=2, padding='same', activation=tf.nn.relu)
        self.bn_5 = tf.keras.layers.BatchNormalization()
        self.maxpool_5 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')

        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()

        self.fc_1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        self.fc_2 = tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)

        self._set_inputs(tf.TensorSpec([None, 80, 1], tf.float32, name='inputs'))


    def call(self, inputs, training=None):
        out = self.conv_1(inputs)
        out = self.bn_1(out, training=training)
        out = self.conv_1_1(out)
        out = self.bn_1_1(out, training=training)
        out = self.maxpool_1(out)

        out = self.conv_2(out)
        out = self.bn_2(out, training=training)
        out = self.conv_2_2(out)
        out = self.bn_2_2(out, training=training)
        out = self.maxpool_2(out)

        out = self.conv_3(out)
        out = self.bn_3(out, training=training)
        out = self.conv_3_3(out)
        out = self.bn_3_3(out, training=training)
        out = self.maxpool_3(out)

        # out = self.conv_4(out)
        # out = self.bn_4(out, training=training)
        # out = self.maxpool_4(out)
        # print(training)
        out = self.avgpool(out)
        out = self.fc_1(out)
        out = self.fc_2(out)

        return out


