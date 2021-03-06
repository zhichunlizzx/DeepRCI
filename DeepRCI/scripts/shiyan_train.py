import os
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers, metrics
from ResNet18 import resnet18
# from mofangDeep import xmm1
from deep_xmm_type04H import xmm1

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"




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
    path_feature_0_train = r'F:\data_xmm\train\0\\'
    path_feature_1_train = r'F:\data_xmm\train\1\\'

    path_feature_00_val = r'F:\data_xmm\validation\0\\'
    path_feature_11_val = r'F:\data_xmm\validation\1\\'
    path_feature_0_val = r'F:\data_xmm\validation\00\\'
    path_feature_1_val = r'F:\data_xmm\validation\11\\'
    path_feature_000_val = r'F:\data_xmm\validation\000\\'
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
    img_dim = 256
    img_channels = 4
    path_train, label_train = path_feature_label(path_feature_1_train, path_feature_0_train)
    path_val_11, label_val_11 = path_feature_label(path_feature_11_val, path_feature_00_val)
    path_val_111, label_val_111 = path_feature_label(path_feature_11_val, path_feature_000_val)
    path_val, label_val = path_feature_label(path_feature_1_val, path_feature_0_val)
    path_val_1,label_val_1 = path_feature_label(path_feature_1_val,path_feature_000_val)
    path_val_100,label_val_100 = path_feature_label(path_val_1_100,path_val_0_100)
    path_val_500, label_val_500 = path_feature_label(path_val_1_500, path_val_0_500)
    path_val_600, label_val_600 = path_feature_label(path_val_1_600, path_val_0_600)
    path_val_700, label_val_700 = path_feature_label(path_val_1_700, path_val_0_700)
    path_val_750, label_val_750 = path_feature_label(path_val_1_750, path_val_0_750)
    path_val_1500, label_val_1500 = path_feature_label(path_val_1_1500, path_val_0_1500)

    # idx = tf.range(235987)
    # idx = tf.random.shuffle(idx)
    #
    # x_train, x_val = tf.gather(path, idx[:180000]), tf.gather(path, idx[180000:])
    # y_train, y_val = tf.gather(label, idx[:180000]), tf.gather(label, idx[180000:])

    # db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = tf.data.Dataset.from_tensor_slices((path_train, label_train))
    db_train = db_train.map(preprocess).shuffle(1000000).batch(12)
    # ab = iter(db_train)
    # print(next(ab))
    # db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    db_val = tf.data.Dataset.from_tensor_slices((path_val, label_val))
    db_val = db_val.map(preprocess).batch(16)

    db_val_11 = tf.data.Dataset.from_tensor_slices((path_val_11, label_val_11))
    db_val_11 = db_val_11.map(preprocess).batch(16)

    db_val_111 = tf.data.Dataset.from_tensor_slices((path_val_111, label_val_111))
    db_val_111 = db_val_111.map(preprocess).batch(16)
    # ab = iter(db_val)
    # print('*'*50)
    # print(next(ab))
    db_val_1 = tf.data.Dataset.from_tensor_slices((path_val_1, label_val_1))
    db_val_1 = db_val_1.map(preprocess).batch(16)

    db_val_100 = tf.data.Dataset.from_tensor_slices((path_val_100, label_val_100))
    db_val_100 = db_val_100.map(preprocess).batch(16)

    db_val_500 = tf.data.Dataset.from_tensor_slices((path_val_500, label_val_500))
    db_val_500 = db_val_500.map(preprocess).batch(16)

    db_val_600 = tf.data.Dataset.from_tensor_slices((path_val_600, label_val_600))
    db_val_600 = db_val_600.map(preprocess).batch(16)

    db_val_700 = tf.data.Dataset.from_tensor_slices((path_val_700, label_val_700))
    db_val_700 = db_val_700.map(preprocess).batch(16)

    db_val_750 = tf.data.Dataset.from_tensor_slices((path_val_750, label_val_750))
    db_val_750 = db_val_750.map(preprocess).batch(16)

    db_val_1500 = tf.data.Dataset.from_tensor_slices((path_val_1500, label_val_1500))
    db_val_1500 = db_val_1500.map(preprocess).batch(16)

    # model = resnet18()
    model = xmm1()
    model.build(input_shape=(None, 256, 256, 4))
    # model.summary()


    optimizer = optimizers.Adam(lr=1e-4)
    # variables = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(200):
        acc_num_train = 0
        num_train = 0
        for step, (path, y) in enumerate(db_train):
            x = img_read(path, img_dim, img_channels)
            # print(x.shape)
            x = tf.convert_to_tensor(x)
            x = tf.cast(x, dtype=tf.float32)
            # print(x)
            z = tf.zeros([14,2])
            with tf.GradientTape() as tape:

                logits = model(x, z, True)

                y_onehot = tf.one_hot(y, depth=2)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
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
        # print('epoch', epoch, 'loss', float(loss))

        # acc_num = 0
        # num = 0
        # y_score = []
        # y_score_11 = []
        # y_score_100 = []
        # y_score_500 = []
        # y_score_600 = []
        # y_score_700 = []
        # y_score_750 = []
        # y_score_1500 = []
        # for path, y in db_val:
        #     # print(path)
        #     x = img_read(path, img_dim, img_channels)
        #     # print(x.shape)
        #     # print(x)
        #     x = tf.convert_to_tensor(x)
        #     x = tf.cast(x, dtype=tf.float32) / 1020.
        #     # print(x.shape)
        #
        #     logits = model(x, False)
        #     prob = tf.nn.softmax(logits, axis=1)
        #     pred = tf.argmax(prob, axis=1)
        #     pred = tf.cast(pred, dtype=tf.int32)
        #     for pro in prob:
        #         y_score.append(pro)
        #     correct = tf.equal(y, pred)
        #     correct = tf.cast(correct, dtype=tf.int32)
        #     correct = tf.reduce_sum(correct)
        #
        #     acc_num += correct
        #     num += x.shape[0]
        #
        # acc_num_val_11 = 0
        # num_val_11 = 0
        #
        # for path, y in db_val_11:
        #     # print(path)
        #     x = img_read(path, img_dim, img_channels)
        #     # print(x.shape)
        #     # print(x)
        #     x = tf.convert_to_tensor(x)
        #     x = tf.cast(x, dtype=tf.float32) / 1020.
        #     # print(x.shape)
        #
        #     logits = model(x, False)
        #     prob = tf.nn.softmax(logits, axis=1)
        #     pred = tf.argmax(prob, axis=1)
        #     pred = tf.cast(pred, dtype=tf.int32)
        #     for pro in prob:
        #         y_score_11.append(pro)
        #
        #     correct = tf.equal(y, pred)
        #     correct = tf.cast(correct, dtype=tf.int32)
        #     correct = tf.reduce_sum(correct)
        #
        #     acc_num_val_11 += correct
        #     num_val_11 += x.shape[0]
        #
        # acc_num_val_111 = 0
        # num_val_111 = 0
        #
        # for path, y in db_val_111:
        #     # print(path)
        #     x = img_read(path, img_dim, img_channels)
        #     # print(x.shape)
        #     # print(x)
        #     x = tf.convert_to_tensor(x)
        #     x = tf.cast(x, dtype=tf.float32) / 1020.
        #     # print(x.shape)
        #
        #     logits = model(x, False)
        #     prob = tf.nn.softmax(logits, axis=1)
        #     pred = tf.argmax(prob, axis=1)
        #     pred = tf.cast(pred, dtype=tf.int32)
        #
        #     correct = tf.equal(y, pred)
        #     correct = tf.cast(correct, dtype=tf.int32)
        #     correct = tf.reduce_sum(correct)
        #
        #     acc_num_val_111 += correct
        #     num_val_111 += x.shape[0]
        #
        # acc_num_val_1 = 0
        # num_val_1 = 0
        #
        # for path, y in db_val_1:
        #     # print(path)
        #     x = img_read(path, img_dim, img_channels)
        #     # print(x.shape)
        #     # print(x)
        #     x = tf.convert_to_tensor(x)
        #     x = tf.cast(x, dtype=tf.float32) / 1020.
        #     # print(x.shape)
        #
        #     logits = model(x, False)
        #     prob = tf.nn.softmax(logits, axis=1)
        #     pred = tf.argmax(prob, axis=1)
        #     pred = tf.cast(pred, dtype=tf.int32)
        #
        #     correct = tf.equal(y, pred)
        #     correct = tf.cast(correct, dtype=tf.int32)
        #     correct = tf.reduce_sum(correct)
        #
        #     acc_num_val_1 += correct
        #     num_val_1 += x.shape[0]
        #
        # acc_num_100 = 0
        # num_100 = 0
        # # y_score = []
        #
        # for path, y in db_val_100:
        #     # print(path)
        #     x = img_read(path, img_dim, img_channels)
        #     # print(x.shape)
        #     # print(x)
        #     x = tf.convert_to_tensor(x)
        #     x = tf.cast(x, dtype=tf.float32) / 1020.
        #     # print(x.shape)
        #
        #     logits = model(x, False)
        #     prob = tf.nn.softmax(logits, axis=1)
        #     pred = tf.argmax(prob, axis=1)
        #     pred = tf.cast(pred, dtype=tf.int32)
        #     for pro in prob:
        #         y_score_100.append(pro)
        #     correct = tf.equal(y, pred)
        #     correct = tf.cast(correct, dtype=tf.int32)
        #     correct = tf.reduce_sum(correct)
        #
        #     acc_num_100 += correct
        #     num_100 += x.shape[0]
        #
        # acc_num_500 = 0
        # num_500 = 0
        # # y_score = []
        # for path, y in db_val_500:
        #     # print(path)
        #     x = img_read(path, img_dim, img_channels)
        #     # print(x.shape)
        #     # print(x)
        #     x = tf.convert_to_tensor(x)
        #     x = tf.cast(x, dtype=tf.float32) / 1020.
        #     # print(x.shape)
        #
        #     logits = model(x, False)
        #     prob = tf.nn.softmax(logits, axis=1)
        #     pred = tf.argmax(prob, axis=1)
        #     pred = tf.cast(pred, dtype=tf.int32)
        #     for pro in prob:
        #         y_score_500.append(pro)
        #     correct = tf.equal(y, pred)
        #     correct = tf.cast(correct, dtype=tf.int32)
        #     correct = tf.reduce_sum(correct)
        #
        #     acc_num_500 += correct
        #     num_500 += x.shape[0]
        #
        # acc_num_600 = 0
        # num_600 = 0
        # # y_score = []
        # for path, y in db_val_600:
        #     # print(path)
        #     x = img_read(path, img_dim, img_channels)
        #     # print(x.shape)
        #     # print(x)
        #     x = tf.convert_to_tensor(x)
        #     x = tf.cast(x, dtype=tf.float32) / 1020.
        #     # print(x.shape)
        #
        #     logits = model(x, False)
        #     prob = tf.nn.softmax(logits, axis=1)
        #     pred = tf.argmax(prob, axis=1)
        #     pred = tf.cast(pred, dtype=tf.int32)
        #     for pro in prob:
        #         y_score_600.append(pro)
        #     correct = tf.equal(y, pred)
        #     correct = tf.cast(correct, dtype=tf.int32)
        #     correct = tf.reduce_sum(correct)
        #
        #     acc_num_600 += correct
        #     num_600 += x.shape[0]
        #
        # acc_num_700 = 0
        # num_700 = 0
        # # y_score = []
        # for path, y in db_val_700:
        #     # print(path)
        #     x = img_read(path, img_dim, img_channels)
        #     # print(x.shape)
        #     # print(x)
        #     x = tf.convert_to_tensor(x)
        #     x = tf.cast(x, dtype=tf.float32) / 1020.
        #     # print(x.shape)
        #
        #     logits = model(x, False)
        #     prob = tf.nn.softmax(logits, axis=1)
        #     pred = tf.argmax(prob, axis=1)
        #     pred = tf.cast(pred, dtype=tf.int32)
        #     for pro in prob:
        #         y_score_700.append(pro)
        #     correct = tf.equal(y, pred)
        #     correct = tf.cast(correct, dtype=tf.int32)
        #     correct = tf.reduce_sum(correct)
        #
        #     acc_num_700 += correct
        #     num_700 += x.shape[0]
        #
        # acc_num_750 = 0
        # num_750 = 0
        # # y_score = []
        # for path, y in db_val_750:
        #     # print(path)
        #     x = img_read(path, img_dim, img_channels)
        #     # print(x.shape)
        #     # print(x)
        #     x = tf.convert_to_tensor(x)
        #     x = tf.cast(x, dtype=tf.float32) / 1020.
        #     # print(x.shape)
        #
        #     logits = model(x, False)
        #     prob = tf.nn.softmax(logits, axis=1)
        #     pred = tf.argmax(prob, axis=1)
        #     pred = tf.cast(pred, dtype=tf.int32)
        #     for pro in prob:
        #         y_score_750.append(pro)
        #     correct = tf.equal(y, pred)
        #     correct = tf.cast(correct, dtype=tf.int32)
        #     correct = tf.reduce_sum(correct)
        #
        #     acc_num_750 += correct
        #     num_750 += x.shape[0]
        #
        #
        # acc_num_1500 = 0
        # num_1500 = 0
        # # y_score = []
        # for path, y in db_val_1500:
        #     # print(path)
        #     x = img_read(path, img_dim, img_channels)
        #     # print(x.shape)
        #     # print(x)
        #     x = tf.convert_to_tensor(x)
        #     x = tf.cast(x, dtype=tf.float32) / 1020.
        #     # print(x.shape)
        #
        #     logits = model(x, False)
        #     prob = tf.nn.softmax(logits, axis=1)
        #     pred = tf.argmax(prob, axis=1)
        #     pred = tf.cast(pred, dtype=tf.int32)
        #     for pro in prob:
        #         y_score_1500.append(pro)
        #     correct = tf.equal(y, pred)
        #     correct = tf.cast(correct, dtype=tf.int32)
        #     correct = tf.reduce_sum(correct)
        #
        #     acc_num_1500 += correct
        #     num_1500 += x.shape[0]
        #
        #
        # train_acc = acc_num_train / num_train
        # acc = acc_num / num
        # acc_val_1 = acc_num_val_1 / num_val_1
        # acc_val_11 = acc_num_val_11 / num_val_11
        # acc_val_111 = acc_num_val_111 / num_val_111
        # acc_100 = acc_num_100 / num_100
        # acc_500 = acc_num_500 / num_500
        # acc_600 = acc_num_600 / num_600
        # acc_700 = acc_num_700 / num_700
        # acc_750 = acc_num_750 / num_750
        # acc_1500 = acc_num_1500 / num_1500
        # print('epoch',epoch,'train_acc:',float(train_acc))
        # print('epoch',epoch, 'fuza acc:',float(acc_val_11))
        # print('epoch', epoch, 'fuza acc 1:', float(acc_val_111))
        # print('epoch:', epoch, 'acc:', float(acc))
        # print('epoch:', epoch, 'acc_val_1:', float(acc_val_1))
        # print('epoch:', epoch, 'acc_100:', float(acc_100))
        # print('epoch:', epoch, 'acc_500:', float(acc_500))
        # print('epoch:', epoch, 'acc_600:', float(acc_600))
        # print('epoch:', epoch, 'acc_700:', float(acc_700))
        # print('epoch:', epoch, 'acc_750:', float(acc_750))
        # print('epoch:', epoch, 'acc_1500:', float(acc_1500))
        # if acc > 0.87:
        #     with open(r'F:\script\y_score_14.txt', 'a') as w_obj:
        #         w_obj.write(str(acc))
        #         w_obj.write('\n')
        #         w_obj.write(str(acc_100))
        #         w_obj.write('\n')
        #         w_obj.write(str(acc_500))
        #         w_obj.write('\n')
        #         w_obj.write(str(acc_600))
        #         w_obj.write('\n')
        #         w_obj.write(str(acc_700))
        #         w_obj.write('\n')
        #         w_obj.write(str(acc_750))
        #         w_obj.write('\n')
        #         w_obj.write(str(acc_1500))
        #         w_obj.write('\n')
        #         y_score = np.asarray(y_score)
        #         y_score_11 = np.asarray(y_score_11)
        #         y_score_100 = np.asarray(y_score_100)
        #         y_score_500 = np.asarray(y_score_500)
        #         y_score_600 = np.asarray(y_score_600)
        #         y_score_700 = np.asarray(y_score_700)
        #         y_score_750 = np.asarray(y_score_750)
        #         y_score_1500 = np.asarray(y_score_1500)
        #         w_obj.write('y_score:')
        #         w_obj.write('\n')
        #         for score in y_score:
        #             w_obj.write(str(score))
        #             w_obj.write('，')
        #         w_obj.write('\n')
        #
        #         w_obj.write('y_score_11:')
        #         w_obj.write('\n')
        #         for score in y_score_11:
        #             w_obj.write(str(score))
        #             w_obj.write('，')
        #         w_obj.write('\n')
        #
        #         w_obj.write('y_score_100:')
        #         w_obj.write('\n')
        #         for score in y_score_100:
        #             w_obj.write(str(score))
        #             w_obj.write('，')
        #         w_obj.write('\n')
        #
        #         w_obj.write('y_score_500:')
        #         w_obj.write('\n')
        #         for score in y_score_500:
        #             w_obj.write(str(score))
        #             w_obj.write('，')
        #         w_obj.write('\n')
        #
        #         w_obj.write('y_score_600:')
        #         w_obj.write('\n')
        #         for score in y_score_600:
        #             w_obj.write(str(score))
        #             w_obj.write('，')
        #         w_obj.write('\n')
        #
        #         w_obj.write('y_score_700:')
        #         w_obj.write('\n')
        #         for score in y_score_700:
        #             w_obj.write(str(score))
        #             w_obj.write('，')
        #         w_obj.write('\n')
        #
        #         w_obj.write('y_score_750:')
        #         w_obj.write('\n')
        #         for score in y_score_750:
        #             w_obj.write(str(score))
        #             w_obj.write('，')
        #         w_obj.write('\n')
        #
        #         w_obj.write('y_score_1500:')
        #         w_obj.write('\n')
        #         for score in y_score_1500:
        #             w_obj.write(str(score))
        #             w_obj.write('，')
        #         w_obj.write('\n')
        # model.save_weights('model.ckpt')
        # if acc_val_1 > 0.9:
        #     model_name = str(acc_val_1)[:3] + 'model.ckpt'
        #     model.save_weights(model_name)
        # if acc>0.83:
        #     model_name = str(acc)[:3] + 'model.ckpt'
        #     model.save_weights(model_name)


if __name__ == '__main__':
    main()