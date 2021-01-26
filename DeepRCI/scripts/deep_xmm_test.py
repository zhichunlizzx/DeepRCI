import os
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers, metrics


def unpicked_patch(file):
    # 打开二进制文件
    # patch_file = open(file,'rb')
    # patch_dict = pickle.load(patch_file, encoding='bytes')
    # print(file)
    # file = str(file)
    img = Image.open(file)
    # img1 = Image.fromarray(np.uint8(img))
    img1 = np.asarray(img)
    return img1
    # return patch_dict


def path_feature_label(dir_path_feature_1, dir_path_feature_0):
    dir_list_1 = os.listdir(dir_path_feature_1)
    path_list_1 = []
    for dir in dir_list_1:
        path = dir_path_feature_1 + dir
        # print(path)
        path_list_1.append(path)

    # for i in range(len(path_list_1)):
    #     # print(path_list_1[i])
    #     a = dir_path_feature_1 + path_list_1[i]
    #     print(a)
    #     path_list_1[i] = a
    #     print(path_list_1[i])

    dir_list_0 = os.listdir(dir_path_feature_0)
    path_list_0 = []
    for dir in dir_list_0:
        path = dir_path_feature_0 + dir
        path_list_0.append(path)
    # for i in range(len(path_list_0)):
    #     path_list_0[i] = dir_path_feature_0 + path_list_0[i]
    # print(path_list_1)
    path_list = np.append(path_list_1, path_list_0)
    # print(type(path_list))
    # print(path_list_0.shape)

    label_1 = np.ones(len(path_list_1))
    label_0 = np.zeros(len(path_list_0))
    label = np.append(label_1, label_0)
    # print(path_list.shape)
    # print(label.shape)
    return path_list, label
    # return path_list


def preprocess(x, y):
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def img_read(path_list, img_dim, img_channels):
    img_feature = np.zeros(shape=(len(path_list), img_dim, img_dim, img_channels))
    # path_list = list(path_list)
    # print(type(path_list))
    path_list = np.asarray(path_list)
    for num, path in enumerate(path_list):
        data = unpicked_patch(path)
        img_feature[num, :, :, :] = data
    img_feature = tf.convert_to_tensor(img_feature)
    img_feature = tf.cast(img_feature, dtype=tf.float32) / 255.
    return img_feature


conv_layers = [  # 5 units of conv + max pooling
    # unit 1
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 2
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 5
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=4, padding='same'),

    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=4, padding='same')

]

fc_layers = [
    # layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(2, activation=None),
]


def main():
    path_feature_0_train = r'F:\data\train\0\\'
    path_feature_1_train = r'F:\data\train\1\\'
    path_feature_0_val = r'F:\data\validation\0\\'
    path_feature_1_val = r'F:\data\validation\1\\'
    img_dim = 256
    img_channels = 4
    path_train, label_train = path_feature_label(path_feature_1_train, path_feature_0_train)
    path_val, label_val = path_feature_label(path_feature_1_val, path_feature_0_val)

    # idx = tf.range(235987)
    # idx = tf.random.shuffle(idx)
    #
    # x_train, x_val = tf.gather(path, idx[:180000]), tf.gather(path, idx[180000:])
    # y_train, y_val = tf.gather(label, idx[:180000]), tf.gather(label, idx[180000:])

    # db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = tf.data.Dataset.from_tensor_slices((path_train, label_train))
    db_train = db_train.map(preprocess).shuffle(1000000).batch(32)
    # ab = iter(db_train)
    # print(next(ab))
    # db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    db_val = tf.data.Dataset.from_tensor_slices((path_val, label_val))
    db_val = db_val.map(preprocess).batch(32)
    # ab = iter(db_val)
    # print('*'*50)
    # print(next(ab))

    conv_net = Sequential(conv_layers)
    conv_net.build(input_shape=[None, 256, 256, 4])

    fc_net = Sequential(fc_layers)
    fc_net.build(input_shape=[None, 512])
    optimizer = optimizers.Adam(lr=1e-5)
    variables = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(200):
        acc_num_train = 0
        num_train = 0
        for step, (path, y) in enumerate(db_train):
            x = img_read(path, img_dim, img_channels)
            # print(x.shape)
            x = tf.convert_to_tensor(x)
            x = tf.cast(x, dtype=tf.float32) / 255.
            # print(x)
            with tf.GradientTape() as tape:
                out = conv_net(x)
                out = tf.reshape(out, [-1, 512])

                logits = fc_net(out)

                y_onehot = tf.one_hot(y, depth=2)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            out_train = conv_net(x)
            out_train = tf.reshape(out, [-1, 512])
            logits = fc_net(out_train)
            prob = tf.nn.softmax(logits)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.equal(y, pred)
            correct = tf.cast(correct, dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            acc_num_train += correct
            num_train += x.shape[0]
            if step % 100 == 0:
                print('epoch:', epoch, 'step:', step, 'loss:', float(loss))
        # print('epoch', epoch, 'loss', float(loss))

        acc_num = 0
        num = 0

        for path, y in db_val:
            # print(path)
            x = img_read(path, img_dim, img_channels)
            # print(x.shape)
            # print(x)
            x = tf.convert_to_tensor(x)
            x = tf.cast(x, dtype=tf.float32) / 255.
            # print(x.shape)
            out = conv_net(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.equal(y, pred)
            correct = tf.cast(correct, dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            acc_num += correct
            num += x.shape[0]

        train_acc = acc_num_train / num_train
        acc = acc_num / num
        print('epoch',epoch,'train_acc:',float(train_acc))
        print('epoch:', epoch, 'acc:', float(acc))


if __name__ == '__main__':
    main()