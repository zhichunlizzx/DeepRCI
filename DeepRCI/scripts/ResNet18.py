import tensorflow as tf


class BasicBlock(tf.keras.layers.Layer):

    # filter_num卷积核的通道数
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        # the first layer
        self.conv1 = tf.keras.layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')

        # the second layer
        self.conv2 = tf.keras.layers.Conv2D(filter_num,(3, 3), strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 短接层
        # 如果stride=1那么shape和原来是保持一致的，这种情况下有没有downsample都行，所以
        # 可以通过if来判断
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x



    def call(self, inputs, training=None):
        # print('layer:', training)
        # [b, h, w, c]
        out = self.conv1(inputs)
        # out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out, training=training)

        identity = self.downsample(inputs)

        output = tf.keras.layers.add([out, identity])

        output = tf.nn.relu(output)

        return output


class ResNet(tf.keras.Model):

    # layer_dims => [2,2,2,2],有四个resblock，每个resblock包含两层basic resblock
    # num_classes 全连接层的输出，取决于有多少类
    def __init__(self, layer_dims, num_classes=2):
        super(ResNet, self).__init__()

        # 预处理层
        self.stem = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (5, 5), strides=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')
        ])

        # 创建四个resblock
        # channel从小到大， feature size从大到小
        self.layer1 = self.build_resblock(32, layer_dims[0],stride=2)
        self.layer2 = self.build_resblock(64, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(128, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(256, layer_dims[3], stride=2)
        self.layer5 = self.build_resblock(512, layer_dims[4], stride=2)

        # output: [b, 512, h, w]

        # 可以把[b, 512, h, w] 变成 [b, 512]
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc0 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.fc1 = tf.keras.layers.Dense(32,activation=tf.nn.relu)





        # 全连接层用来分类
        self.fc = tf.keras.layers.Dense(num_classes)



    def call(self, inputs, training=None):
        # print('net:', training)
        x = self.stem(inputs)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.layer5(x, training=training)
        # print(x.shape)
        # x = self.avgpool(x)
        x = tf.reshape(x, [-1, 2048])
        x = self.fc0(x)
        x = self.fc1(x)

        # [b,2]
        x = self.fc(x)
        return x


    # blocks 当前网络会堆叠多少个resblock
    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = tf.keras.Sequential()

        # stride等于传入的参数这个stride的话，如果传入的不为1 就相当于做了下采样

        res_blocks.add(BasicBlock(filter_num, stride))

        # 后续的resblock不再进行下采样
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


def resnet18():

    return ResNet([2, 3, 5, 3, 2])

def resnet34():

    return ResNet([3, 4, 6, 3])

