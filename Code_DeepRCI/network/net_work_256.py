import tensorflow as tf


class XmmNet0(tf.keras.Model):

    def __init__(self):
        super(XmmNet0, self).__init__()

        # unit 1
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        # unit 2
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        # unit 3
        self.conv5 = tf.keras.layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.conv6 = tf.keras.layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        # unit 4
        self.conv7 = tf.keras.layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.conv8 = tf.keras.layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.bn8 = tf.keras.layers.BatchNormalization()
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        # unit 5
        self.conv9 = tf.keras.layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.bn9 = tf.keras.layers.BatchNormalization()
        self.conv10 = tf.keras.layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.bn10 = tf.keras.layers.BatchNormalization()
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        # self.conv11 = tf.keras.layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        # self.bn11 = tf.keras.layers.BatchNormalization()
        # self.conv12 = tf.keras.layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        # self.bn12 = tf.keras.layers.BatchNormalization()
        # self.pool6 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        '''
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=[3, 3], strides=2,activation=tf.nn.relu, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=[3, 3], strides=2, activation=tf.nn.relu, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=2, padding='same')

        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=[3, 3], strides=2, activation=tf.nn.relu, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[3,3], strides=2, padding='same')

        self.conv4 = tf.keras.layers.Conv2D(128, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()


        self.conv5 = tf.keras.layers.Conv2D(256, kernel_size=[3, 3], strides=2, activation=tf.nn.relu, padding='same')
        self.bn5 = tf.keras.layers.BatchNormalization()

        self.conv6 = tf.keras.layers.Conv2D(384, kernel_size=[3, 3], strides=2, activation=tf.nn.relu, padding='same')
        self.bn6 = tf.keras.layers.BatchNormalization()
        '''
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

        self.fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        # self.drop1 = tf.nn.dropout(ra)
        # self.dr1 = tf.keras.layers.Dropout(rate=0.5)
        self.fc3 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        # self.dr2 = tf.keras.layers.Dropout(rate=0.5)
        self.fc4 = tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)

    def call(self, inputs, training=None):
        # if_train = True
        # print(training)
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.pool1(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)
        out = self.conv4(out)
        out = self.bn4(out, training=training)
        out = self.pool2(out)

        out = self.conv5(out)
        out = self.bn5(out, training=training)
        out = self.conv6(out)
        out = self.bn6(out, training=training)
        out = self.pool3(out)

        out = self.conv7(out)
        out = self.bn7(out, training=training)
        out = self.conv8(out)
        out = self.bn8(out, training=training)
        out = self.pool4(out)
        #
        out = self.conv9(out)
        out = self.bn9(out, training=training)
        out = self.conv10(out)
        out = self.bn10(out, training=training)
        out = self.pool5(out)
        #
        # out = self.conv11(out)
        # out = self.bn11(out, training=training)
        # out = self.conv12(out)
        # out = self.bn12(out, training=training)
        # out = self.pool6(out)

        # out = self.conv5(out)
        # out = self.bn5(out, training=training)
        #
        # out = self.conv6(out)
        # out = self.bn6(out, training=training)

        out = self.avgpool(out)
        # out = tf.reshape(out, [-1, 8192])
        # out = self.fc1(out)
        out = self.fc2(out)
        # out = self.dr1(out, training=training)
        out = self.fc3(out)
        # out = self.dr2(out, training=training)
        out = self.fc4(out)

        return out


def xmm1():
    return XmmNet0()



if __name__ == '__main__':
    a = XmmNet0()
    print(a.summary())