import tensorflow as tf
from net_work_256 import xmm1
import numpy as np
from PIL import Image
import sys

def unpicked_patch(file):
    # 打开二进制文件
    img = Image.open(file)
    img1 = np.asarray(img)
    return img1


def img_read(path_list, img_dim, img_channels):
    img_feature = np.zeros(shape=(len(path_list), img_dim, img_dim, img_channels))
    path_list = np.asarray(path_list)
    for num, path in enumerate(path_list):
        data = unpicked_patch(path)
        img_feature[num, :, :, :] = data
    img_feature = tf.convert_to_tensor(img_feature)
    img_feature = tf.cast(img_feature, dtype=tf.float32)
    return img_feature


def main():
    img_dim = 448
    img_channel = 3

    model = xmm1()
    model.build(input_shape=(None, 448, 448, 3))
    model.load_weights('model.ckpt')
    # model.summary()

    # change to manual input later
    input_path = [r'C:\dataz\research\DeepRCI\data\1\A0A0A0LJF4.00.jpg']
    img_feature = img_read(input_path, img_dim, img_channel)

    output = model(img_feature)
    output = tf.argmax(output, axis=1)
    print(list(output.numpy()))



if __name__ == '__main__':
    main()