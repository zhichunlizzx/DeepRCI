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
import tensorboard
import datetime
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
        img_feature[num, :, :, :] = data
    img_feature = tf.convert_to_tensor(img_feature)
    img_feature = tf.cast(img_feature, dtype=tf.float32)
    return img_feature







def main():
    path_feature_0_train = r'H:\400\train_balance\0\\'
    path_feature_1_train = r'H:\400\train\1\\'

    path_feature_0_val = r'E:\my_research\DeepRCI\duli_ceshi\0\\'
    path_feature_1_val = r'E:\my_research\DeepRCI\duli_ceshi\abc\\'


    path_feature_test_1 = r'H:\400\test\1/'
    path_feature_test_0 = r'H:\400\test\0/'

    img_dim = 400
    img_channels = 4
    path_train, label_train = path_feature_label(path_feature_1_train, path_feature_0_train)
    path_val, label_val = path_feature_label(path_feature_1_val, path_feature_0_val)
    path_test, label_test = path_feature_label(path_feature_test_1, path_feature_test_0)


    db_train = tf.data.Dataset.from_tensor_slices((path_train, label_train))
    db_train = db_train.map(preprocess).shuffle(1000000).batch(8)

    db_val = tf.data.Dataset.from_tensor_slices((path_val, label_val))
    db_val = db_val.map(preprocess).batch(16)

    db_test = tf.data.Dataset.from_tensor_slices((path_test, label_test))
    db_test = db_test.map(preprocess).batch(16)


    # model = resnet18()
    model = xmm1()
    model.build(input_shape=(None, 400, 400, 4))
    # model.summary()
    learning_r = 1e-4

    optimizer = optimizers.Adam(lr=learning_r)
    # variables = conv_net.trainable_variables + fc_net.trainable_variables


    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = r'H:\400\logs/' + current_time + '_12'
    # summary_writer = tf.summary.create_file_writer(log_dir)
    for epoch in range(200):
        acc_num_train = 0
        num_train = 0
        loss_train = 0
        for step, (path, y) in enumerate(db_train):
            x = img_read(path, img_dim, img_channels)
            # print(x.shape)
            x = tf.convert_to_tensor(x)
            x = tf.cast(x, dtype=tf.float32)
            # print(x)
            with tf.GradientTape() as tape:

                logits = model(x, True)

                y_onehot = tf.one_hot(y, depth=2)
                # tf.losses.binary_crossentropy
                loss = tf.reduce_mean(tf.losses.binary_crossentropy(y_onehot, logits, from_logits=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # logits_train = model(x, training = )
            prob = tf.nn.softmax(logits)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.equal(y, pred)
            correct = tf.cast(correct, dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            acc_num_train += correct
            num_train += x.shape[0]

            if step % 1000 == 0:
                print('epoch:', epoch, 'step:', step, 'loss:', float(loss))
            if step > 1:
                loss_train += loss
                loss_train = loss_train / 2
            # with summary_writer.as_default():
            #     tf.summary.scalar("loss step" + str(epoch), loss, step=step)
        # print('epoch', epoch, 'loss', float(loss))

        acc_num = 0
        num = 0




        numm = 0
        for path, y in db_val:
            # print(path)
            x = img_read(path, img_dim, img_channels)
            # print(x.shape)
            # print(x)
            x = tf.convert_to_tensor(x)
            x = tf.cast(x, dtype=tf.float32)
            # print(x.shape)

            logits = model(x, False)

            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.equal(y, pred)
            correct = tf.cast(correct, dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            acc_num += correct
            num += x.shape[0]




        acc_num_test = 0
        num_test = 0

        for path, y in db_test:
            # print(path)
            x = img_read(path, img_dim, img_channels)
            # print(x.shape)
            # print(x)
            x = tf.convert_to_tensor(x)
            x = tf.cast(x, dtype=tf.float32)
            # print(x.shape)

            logits = model(x, False)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.equal(y, pred)
            correct = tf.cast(correct, dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            acc_num_test += correct
            num_test += x.shape[0]




        train_acc = acc_num_train / num_train
        acc = acc_num / num
        acc_test = acc_num_test / num_test

        print('epoch',epoch,'train_acc:',float(train_acc))

        print('epoch:', epoch, 'acc:', float(acc))
        print('epoch', epoch, 'acc_test:', float(acc_test))

        # with summary_writer.as_default():
        #     tf.summary.scalar("epoch_train_loss", loss_train, step=epoch)
        #     tf.summary.scalar('epoch_train_acc', train_acc, step=epoch)
        #     # tf.summary.scalar('epoch_val_loss', loss_val, step=epoch)
        #     tf.summary.scalar('epoch_val_acc', acc, step=epoch)


        if epoch  == 10:
            learning_r = learning_r / 10
        elif epoch > 10:
            if (epoch % 5) == 0:
                learning_r = learning_r / 10
        optimizer = optimizers.Adam(lr=learning_r)




if __name__ == '__main__':
    main()