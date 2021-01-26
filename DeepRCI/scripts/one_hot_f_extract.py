from Bio import SeqIO
import numpy as np
amino_acid = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8,
              'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16,
              'V':17, 'W':18, 'Y':19}
sequence = {}

def one_hot(section_seq):
    """
    3-gram 三个氨基酸对应的编码乘起来的数n，那么这个1*8000的矩阵的第n个位置，赋值为1,
    但是0怎么办—这个方法不行 1*1*2 2*1*1相同

    将给出的数据表示成one-hot编码
    :param feau:
    :return: ont-hot
    给定数据的one-hot表示形式
    """
    one_hot = np.zeros(8000)
    loc_in_oht = amino_acid[section_seq[0]]*400+amino_acid[section_seq[1]]*20+amino_acid[section_seq[2]]
    one_hot[loc_in_oht] = 1
    print(one_hot)
    print(list(one_hot))
    print(list(one_hot).index(1))
    return one_hot

def tri_gram(seq):
    """
    计算序列的3gram特征
    :param seq:
    :return:
    """
    tri_gram_matrix = []
    for i in range(len(seq)-2):
        #将3-gram表示成one-hot编码
        print(seq[i:i+3])
        # one_hot(seq[i:i+3])
        one_hot_r = one_hot(seq[i:i+3])
        tri_gram_matrix.append(one_hot_r)

    #将结果返回，
    return tri_gram_matrix
    pass


def sequence_ex(path):
    """
    通过biopython提取fasta文件中的序列及标签信息
    :return: None
    """
    for seq_record in SeqIO.parse(path, 'fasta'):
        print(seq_record.id)
        print(seq_record.seq)
        print(len(seq_record.seq))
        sequence[seq_record.id] = seq_record.seq
        tri_maxtrix = tri_gram(seq_record.seq)
        print(tri_maxtrix)
        for arr in tri_maxtrix:
            print(list(arr).index(1))

    return None


if __name__ == '__main__':
    fas_path = r'E:\111.seq'
    #fas_path = input("请输入fasta文件所在路径")
    # id, seq = sequence_ex(fas_path)
    sequence_ex(fas_path)

    # print(id)
    # sequence = {}
    # sequence[id] = seq
    # print(sequence[id])
    # print(sequence)

    # one_hot('AYY')
    pass