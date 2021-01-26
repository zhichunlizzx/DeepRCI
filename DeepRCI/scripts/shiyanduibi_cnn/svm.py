import pandas as pd
import numpy as np
from sklearn.svm import SVC

def get_feature(path, num):
    csv_data = pd.read_csv(path, engine='python')
    csv_data = np.array(csv_data)
    return csv_data[:,:num], csv_data[:,num]

def main():
    svm_class = SVC(C=10000, gamma='auto', kernel='rbf')

    path_train = r'E:\data\AAC\one_gram\twogram.csv'
    path_test = r'E:\data\AAC\one_gram\twogram_val.csv'

    fearure_train, label_train = get_feature(path_train, 400)
    fearure_test, label_test = get_feature(path_test, 400)

    a = svm_class.fit(fearure_train, label_train)
    y_score = a.decision_function(fearure_test)
    pre = a.predict(fearure_test)
    num = 0
    pos = 0
    for i in range(len(pre)):
        if pre[i] == label_test[i]:
            num += 1
            if pre[i] == 1:
                pos += 1
    # print(num)
    print(len(y_score))
    print(list(label_test).count(1))
    print('SVM')
    print(num / len(pre))
    print(num)
    print(pos)


if __name__ == '__main__':
    main()