import os
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers, metrics
from ResNet18 import resnet18
# from mofangDeep import xmm1
from deep_xmm_type04G import xmm1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




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
    # path_list = path_list_1
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
        data = np.array(data)
        print(path)
        img_feature[num, :, :, :] = data

    img_feature = tf.convert_to_tensor(img_feature)
    img_feature = tf.cast(img_feature, dtype=tf.float32)
    return img_feature







def main():
    path_feature_0_train = r'H:\400\train\0\\'
    path_feature_1_train = r'H:\400\train\1\\'

    path_feature_00_val = r'F:\data_xmm\validation\0\\'
    path_feature_11_val = r'F:\data_xmm\validation\1\\'
    path_feature_0_val = r'H:\400\validation\0\\'
    path_feature_1_val = r'H:\400\validation\1\\'
    path_feature_000_val = r'H:\400\validation\00\\'
    path_feature_one_1 = r'H:\400\validation_1\1\\'
    path_feature_one_0 = r'H:\400\validation_1\0\\'
    path_feature_one_000 = r'H:\400\validation_1\00\\'
    path_val_1_100 = r'F:\data_xmm\100\1\\'
    path_val_0_100 = r'F:\data_xmm\100\0\\'
    path_val_1_500 = r'F:\data_xmm\500\1\\'
    path_val_0_500 = r'F:\data_xmm\500\0\\'
    path_val_1_600 = r'F:\data_xmm\600\1\\'
    path_val_0_600 = r'F:\data_xmm\600\0\\'
    path_val_1_700 = r'F:\data_xmm\700\1\\'
    path_val_0_700 = r'F:\data_xmm\700\0\\'
    path_val_1_750 = r'F:\data_xmm\750\1\\'
    path_val_0_750 = r'F:\data_xmm\750\0\\'
    path_val_1_1500 = r'F:\data_xmm\100-500\1\\'
    path_val_0_1500 = r'F:\data_xmm\100-500\0\\'
    img_dim = 400
    img_channels = 4
    path_train, label_train = path_feature_label(path_feature_1_train, path_feature_0_train)
    # path_val_11, label_val_11 = path_feature_label(path_feature_11_val, path_feature_00_val)
    # path_val_111, label_val_111 = path_feature_label(path_feature_11_val, path_feature_000_val)
    path_val, label_val = path_feature_label(path_feature_1_val, path_feature_0_val)
    path_val_1, label_val_1 = path_feature_label(path_feature_1_val, path_feature_000_val)

    #1
    path_val_one, label_val_one = path_feature_label(path_feature_one_1, path_feature_one_0)
    path_val_one_1, label_val_one_1 = path_feature_label(path_feature_one_1, path_feature_one_000)
    # path_val_100,label_val_100 = path_feature_label(path_val_1_100,path_val_0_100)
    # path_val_500, label_val_500 = path_feature_label(path_val_1_500, path_val_0_500)
    # path_val_600, label_val_600 = path_feature_label(path_val_1_600, path_val_0_600)
    # path_val_700, label_val_700 = path_feature_label(path_val_1_700, path_val_0_700)
    # path_val_750, label_val_750 = path_feature_label(path_val_1_750, path_val_0_750)
    # path_val_1500, label_val_1500 = path_feature_label(path_val_1_1500, path_val_0_1500)

    # idx = tf.range(235987)
    # idx = tf.random.shuffle(idx)
    #
    # x_train, x_val = tf.gather(path, idx[:180000]), tf.gather(path, idx[180000:])
    # y_train, y_val = tf.gather(label, idx[:180000]), tf.gather(label, idx[180000:])

    # db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = tf.data.Dataset.from_tensor_slices((path_train, label_train))
    db_train = db_train.map(preprocess).shuffle(1000000).batch(8)
    # ab = iter(db_train)
    # print(next(ab))
    # db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    db_val = tf.data.Dataset.from_tensor_slices((path_val, label_val))
    db_val = db_val.map(preprocess).batch(8)

    # db_val_11 = tf.data.Dataset.from_tensor_slices((path_val_11, label_val_11))
    # db_val_11 = db_val_11.map(preprocess).batch(16)
    #
    # db_val_111 = tf.data.Dataset.from_tensor_slices((path_val_111, label_val_111))
    # db_val_111 = db_val_111.map(preprocess).batch(16)
    # ab = iter(db_val)
    # print('*'*50)
    # print(next(ab))
    db_val_1 = tf.data.Dataset.from_tensor_slices((path_val_1, label_val_1))
    db_val_1 = db_val_1.map(preprocess).batch(8)

    db_val_one = tf.data.Dataset.from_tensor_slices((path_val_one, label_val_one))
    db_val_one = db_val_one.map(preprocess).batch(8)

    db_val_one_1 = tf.data.Dataset.from_tensor_slices((path_val_one_1, label_val_one_1))
    db_val_one_1 = db_val_one_1.map(preprocess).batch(8)

    # db_val_100 = tf.data.Dataset.from_tensor_slices((path_val_100, label_val_100))
    # db_val_100 = db_val_100.map(preprocess).batch(16)
    #
    # db_val_500 = tf.data.Dataset.from_tensor_slices((path_val_500, label_val_500))
    # db_val_500 = db_val_500.map(preprocess).batch(16)
    #
    # db_val_600 = tf.data.Dataset.from_tensor_slices((path_val_600, label_val_600))
    # db_val_600 = db_val_600.map(preprocess).batch(16)
    #
    # db_val_700 = tf.data.Dataset.from_tensor_slices((path_val_700, label_val_700))
    # db_val_700 = db_val_700.map(preprocess).batch(16)
    #
    # db_val_750 = tf.data.Dataset.from_tensor_slices((path_val_750, label_val_750))
    # db_val_750 = db_val_750.map(preprocess).batch(16)
    #
    # db_val_1500 = tf.data.Dataset.from_tensor_slices((path_val_1500, label_val_1500))
    # db_val_1500 = db_val_1500.map(preprocess).batch(16)

    # model = resnet18()
    model = xmm1()
    model.build(input_shape=(None, 400, 400, 4))
    # model.summary()


    optimizer = optimizers.Adam(lr=1e-4)
    # variables = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(200):
        acc_num_train = 0
        num_train = 0
        for step, (path, y) in enumerate(db_train):
            print(path)
            print(step)
            x = img_read(path, img_dim, img_channels)
            print(x.shape)
        # for path, y in db_val_one:
        #     print(path)
        #
        #     x = img_read(path, img_dim, img_channels)
        #     print(x.shape)
        #     print(epoch)





if __name__ == '__main__':
    main()
    # x = img_read([b'H:\\400\\validation\\H3GAN9\\\\H3GAN9.05.jpg'], 400, 4)
    # print(x)