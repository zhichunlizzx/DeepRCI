import numpy as np
import pandas as pd
from network import CMM_NET
import tensorflow as tf

def get_feature(path, num):
    csv_data = pd.read_csv(path, engine='python')
    csv_data = np.array(csv_data)
    return csv_data[:,:num], csv_data[:,num]

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y



def main():
    path_train = r'E:\data\PSSM\train\train.csv'
    path_test = r'E:\data\PSSM\val\val.csv'

    feature_train, label_train = get_feature(path_train, 20)
    feature_test, label_test = get_feature(path_test, 20)

    feature_train = tf.expand_dims(feature_train, axis=2)
    feature_test = tf.expand_dims(feature_test, axis=2)

    db_train = tf.data.Dataset.from_tensor_slices((feature_train, label_train))
    db_train = db_train.map(preprocess).shuffle(100000).batch(32)

    db_test = tf.data.Dataset.from_tensor_slices((feature_test, label_test))
    db_test = db_test.map(preprocess).batch(32)

    model = CMM_NET()

    learning_rate = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(2000):
        acc_num_train = 0
        num_train = 0
        # loss_train = 0
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                logits = model(x, True)
                y_onehot = tf.one_hot(y, depth=2)
                loss = tf.reduce_mean(tf.losses.binary_crossentropy(y_onehot, logits, from_logits=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            prob = tf.nn.softmax(logits)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.equal(y, pred)
            correct = tf.cast(correct, dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            acc_num_train += correct
            num_train += x.shape[0]

        acc_num = 0
        num = 0
        num_acc_pos = 0
        num_pos = 0
        y_score = []
        for x, y in db_test:
            logits = model(x, False)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            for pro in prob:
                y_score.append(pro)

            correct = tf.equal(y, pred)
            correct = tf.cast(correct, dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            acc_num += correct
            num += x.shape[0]

            for i in range(len(y)):
                if y[i] == 1:
                    num_pos += 1
                    if y[i] == pred[i]:
                        num_acc_pos += 1

        train_acc = acc_num_train / num_train
        acc = acc_num / num
        acc_pos = num_acc_pos / num_pos
        print('epoch', epoch, 'train_acc:', float(train_acc))
        print('epoch:', epoch, 'acc:', float(acc))
        print('epoch:', epoch, 'acc_pos:', float(acc_pos))

        if acc >= 0.938:
            with open(r'H:\400\y_score_pssm.txt', 'a') as w_obj:
                w_obj.write(str(acc))
                w_obj.write('\n')
                y_score = np.asarray(y_score)
                w_obj.write('y_score:')
                w_obj.write('\n')
                for score in y_score:
                    w_obj.write(str(score))
                    w_obj.write('，')
                w_obj.write('\n')

if __name__ == '__main__':
    main()