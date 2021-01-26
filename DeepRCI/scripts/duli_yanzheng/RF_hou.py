from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from sklearn.decomposition import PCA

def get_feature(path, num):
    csv_data = pd.read_csv(path, engine='python')
    csv_data = np.array(csv_data)
    # print(csv_data.shape)
    return csv_data[:,:num], csv_data[:,num]

def main():
    path_trian = r'E:\my_research\DeepRCI\duli_ceshi\atp_train.csv'
    path_test = r'E:\my_research\DeepRCI\duli_ceshi\pos_test.csv'

    feature_train, label_train = get_feature(path_trian, 188)
    feature_test, label_test = get_feature(path_test, 188)

    feature = np.vstack((feature_train, feature_test))
    pca = PCA(n_components=20)
    feature = pca.fit_transform(feature)
    feature_train, feature_test = feature[:7998], feature[7998:]




    # pca = PCA(n_components=2)
    # feature = pca.fit_transform(feature)
    # feature_test = pca.fit_transform(feature_test)




    # print(feature_train)
    # print(label_train)
    RF = RandomForestClassifier(random_state=15)
    classifier = RF.fit(feature_train, label_train)
    # y_score = classifier.predict_proba(feature_test)

    pre = classifier.predict(feature_test)
    num = 0
    pos = 0
    for i in range(len(pre)):
        if pre[i] == label_test[i]:
            num += 1
            if pre[i] == 1:
                pos += 1
    print('RF')
    print(num / len(pre))
    print(num)
    print(pos)

if __name__ == '__main__':
    main()