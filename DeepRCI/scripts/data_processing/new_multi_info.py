from Bio_fasta import ex_seq_from_fasta
import math
import numpy as np
import xlwt
import xlsxwriter

amino_acid = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
              'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
              'V', 'W', 'Y']
pseudo_count = 1


def calculate_ma(M, L, id_seq):
    """
    返回MSA的全部ma值
    :param M: MSA的序列个数
    :param L: 序列的长度
    :param id_seq: 多序列比对
    :return: ma_value 存储着每条序列的ma值
    """
    x = 0.9
    ma_value = []
    for m in range(M):
        b = 0
        for i in range(M):
            sum = 0
            # 这里有原来的range(i)改成了range(M)
            for j in range(L):
                if id_seq[m][j] == id_seq[i][j]:
                    sum += 1
            if sum >= (x * L):
                b += 1
        ma_value.append(b)

    return ma_value
    pass


def fi_score(M, i, a_a_A, id_seq, ma_value):
    """
    计算fi(a_a)，并返回
    :param M:多序列比对的行数
    :param i:所要计算的行
    :param a_a:当前计算的氨基酸amino acid
    :param id_seq:多序列比对，二维列表
    :param ma_value:ma值的一个列表
    :return:fi(a_a)
    """
    sum = 0
    Meff = 0
    for m in range(M):
        # 其实能改成先把所有ma算出来，但是存储和计算也是问题
        # ma = calculate_ma(M, L, m, id_seq)
        ma = ma_value[m]
        if id_seq[m][i] == a_a_A:
            sum += 1/ma
        Meff += 1/ma
        # print(sum)
    # 待修改
    fi = (sum + pseudo_count/21) / (pseudo_count+Meff)

    return fi


def fij_score(M, i, j, a_a_A, a_a_B, id_seq, ma_value):
    """
    计算i，j列上残基A和残基B同时出现的频率
    :param M: MSA的序列个数
    :param i:第i列
    :param j:第j列
    :param a_a_A:残基A
    :param a_a_B:残基B
    :param id_seq:多序列比对
    :param ma_value:各个序列的ma值
    :return:fij
    """
    sum = 0
    Meff = 0
    # print(M)
    for m in range(M):
        ma = ma_value[m]
        if (id_seq[m][i] == a_a_A) & (id_seq[m][j] == a_a_B):
            sum += 1/ma
        # print(sum)
        Meff += 1/ma
    # 待修改
    fij = (sum + pseudo_count/(21*21)) / (Meff + pseudo_count)
    return fij


def multi_information(path):
    file_path_excel = r'E:\data\result\MI.xls'
    workbook = xlsxwriter.Workbook(file_path_excel)

    sheet1 = workbook.add_worksheet('MI')

    id_seq = ex_seq_from_fasta(path)
    # print(id_seq)
    print(len(id_seq))
    # alignments = []
    # for id,seq in id_seq.items():
    #     alignments.append(seq)
    #     print("alignment had append the sequence %s" %id)
    # print(len(alignments))
    M = len(id_seq)
    L = len(id_seq[0])
    q = 21
    MI = np.zeros([L, L])
    ma_value = calculate_ma(M, L ,id_seq)
    print(ma_value)
    # 还能再优化一下，只考虑半个矩阵，因为是对角的啊
    test_sum = 0
    # 只计算一半矩阵
    for i in range(L):
        for j in range(i+1):
            sum_ab = 0
            for a_a_A in amino_acid:
                fi_value = fi_score(M, i, a_a_A, id_seq, ma_value)
                # print(fi_value)
                for a_a_B in amino_acid:
                    fj_value = fi_score(M, j, a_a_B, id_seq,ma_value)
                    fij_value = fij_score(M, i, j, a_a_A, a_a_B, id_seq, ma_value)

                    # print(fj_value)
                    # print(fij_value)
                    sum_ab += fij_value * (math.log((fij_value/(fi_value * fj_value)),math.e))
                    # print(sum_ab)
            # MI_ij = sum_ab
            MI[i][j] = sum_ab
            sheet1.write(i, j , sum_ab)
            MI[j][i] = sum_ab
            sheet1.write(j, i, sum_ab)
            print('-' * 50)
            print(sum_ab)
            test_sum +=1
            print("test_sum:",test_sum)
            print(i,j)
    workbook.close()

if __name__ == '__main__':
    # file_path = input("please input the path of the align file(fasta):")
    file_path = r'E:\data\result\two.fas'
    multi_information(file_path)
    pass
