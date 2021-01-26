import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def str_int(num):
    if num == '1':
        return 1
    if num == '0':
        return 0

def yy_label(row):
    with open(r'E:\my_research\transporter_svm\roc.txt', 'r') as r_obj:
        y_label = r_obj.readlines()
    y_label = y_label[row]
    y_label = y_label[:-1]
    y_label = y_label.split(',')
    y_label = y_label[:-1]
    a_label = []
    # print(y_label)
    # for b in y_label:
    #     print(b)
    for i in range(len(y_label)):
        # y_label[i] = int(y_label[i])
        a_label.append(str_int(y_label[i]))
    return a_label


def y_score(path, row):
    with open(path, 'r') as r_obj:
    # with open(r'H:\400\y_score_12_512.txt', 'r') as r_obj:
        y_score = r_obj.readlines()
    y_score = y_score[row]
    # print(y_score)
    y_score = y_score[:-2]
    # y_score = y_score.split('，')
    y_score = y_score.split('，')
    y_score_87 = []
    for score in y_score:
        score = score.split(' ')
        # score = score[1]
        while score[-1] == ']':
            score.pop()
        while score[-1] == '':
            score.pop()
        score = score[-1]
        if score[-1] == ']':
            score = score[:-1]
        score = float(score)
        y_score_87.append(score)
    return y_score_87
# print(len(y_score_87))

def y_label(pos, neg):
    y_label = []
    for i in range(0, pos):
        y_label.append(1.0)
    for i in range(0, neg):
        y_label.append(0.0)
    return y_label


def main():

    # RCI
    path_RCI = r'H:\400\y_score_12_512.txt'
    y_score_RCI = y_score(path_RCI, 5)
    y_label_RCI = y_label(500, 500)
    fpr_RCI, tpr_RCI, threshold = roc_curve(y_label_RCI, y_score_RCI)
    roc_auc_RCI = auc(fpr_RCI, tpr_RCI)

    # PSSM
    path_PSSM = r'H:\400\y_score_pssm.txt'
    y_score_PSSM = y_score(path_PSSM, 5)
    y_label_PSSM = y_label(498, 500)
    fpr_PSSM, tpr_PSSM, threshold = roc_curve(y_label_PSSM, y_score_PSSM)
    roc_auc_PSSM = auc(fpr_PSSM, tpr_PSSM)

    # SS
    path_SS = r'H:\400\y_score_ss.txt'
    y_score_SS = y_score(path_SS, 374)
    y_label_SS = y_label(500, 499)
    fpr_SS, tpr_SS, threshold = roc_curve(y_label_SS, y_score_SS)
    roc_auc_SS = auc(fpr_SS, tpr_SS)

    # one_gram
    path_one = r'H:\400\y_score_onegram.txt'
    y_score_one = y_score(path_one, 41)
    y_label_one = y_label(500, 500)
    fpr_one, tpr_one, threshold = roc_curve(y_label_one, y_score_one)
    roc_auc_one = auc(fpr_one, tpr_one)

    # two_gram
    path_two = r'H:\400\y_score_twogram.txt'
    y_score_two = y_score(path_two, 14)
    y_label_two = y_label(499, 500)
    fpr_two, tpr_two, threshold = roc_curve(y_label_two, y_score_two)
    roc_auc_two = auc(fpr_two, tpr_two)

    # t_gram
    path_t = r'H:\400\y_score_t_gram.txt'
    y_score_t = y_score(path_t, 8)
    y_label_t = y_label(499, 500)
    fpr_t, tpr_t, threshold = roc_curve(y_label_t, y_score_t)
    roc_auc_t = auc(fpr_t, tpr_t)


    plt.figure()
    lw = 1
    plt.figure(figsize=(10, 10))
    plt.plot(fpr_RCI, tpr_RCI, color='black',
             lw=lw, label='RCI (AUC = %0.2f)' % roc_auc_RCI)
    plt.plot(fpr_PSSM, tpr_PSSM, color='darkblue',
             lw=lw, label='PSSM (AUC = %0.2f)' % roc_auc_PSSM)
    plt.plot(fpr_SS, tpr_SS, color='darkviolet',
             lw=lw, label='SS (AUC = %0.2f)' % roc_auc_SS)
    plt.plot(fpr_one, tpr_one, color='green',
             lw=lw, label='1-gram (AUC = %0.2f)' % roc_auc_one)
    plt.plot(fpr_two, tpr_two, color='darkorange',
             lw=lw, label='2-gram (AUC = %0.2f)' % roc_auc_two)
    plt.plot(fpr_two, tpr_two, color='red',
             lw=lw, label='3-gram (AUC = %0.2f)' % roc_auc_t)



    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.show()



if __name__ == '__main__':
    main()