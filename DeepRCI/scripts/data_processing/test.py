from Bio import SeqIO
from browsermobproxy import Server
import sample_pad_ccmpred_to_image
import numpy as np
import matplotlib.pyplot as plt
def test_of_0005524(path):
    sum = 0
    with open(path, mode='r') as r_obj:
        lines = r_obj.readlines()
        r_obj.close()
    for line in lines:
        print(line)
        if '0005524' in line:
            continue
        sum += 1
    print(sum)


# 统计fasta数量
def ex_seq_from_fasta(path):
    # fasta = {}
    fasta = []
    sum = 0
    for seq_data in SeqIO.parse(path, 'fasta'):
        fasta.append(seq_data.seq)
        # fasta_1.append(seq_data.id)
        # print(seq_data.name)
        sum = sum+1
    # for id in fasta:
    #     sum1 = 0
    #     print(id)
    #     for id1 in fasta_1:
    #         if id == id1:
    #             sum1 = sum1+1
    print(sum)
    print(sum)
    return sum


def network_spider():
    url = 'https://www.ebi.ac.uk/QuickGO/annotations?geneProductId=T1FNQ7'



if __name__ == '__main__':
    path1 = r'H:\data\0005524\mat_0005524\B1N4C3.mat'
    path = r'H:\data\0005524\zhangzhaoxi222.jpg'

    a = np.loadtxt(path1)
    plt.imsave(path, a , cmap=plt.cm.gray_r)
    # sample_pad_ccmpred_to_image(path1, path)

    #print(ex_seq_from_fasta(path))
    # test_of_0005524(path1)
    # a = "0005524"
    # b = a.split('|')
    # print(b)
    # if '0005524' in 'GO:'0005524':

    # print('0005524' in 'GO:0005524')









# import math
# import xlsxwriter
# #print(math.log(10,math.e))
# file_path_excel = r'E:\data\result\MI.xlsx'
# workbook = xlsxwriter.Workbook(file_path_excel)
# worksheet = workbook.add_worksheet("MI")
# for i in range(260):
#     worksheet.write(i,0,"aaa")
# workbook.close()
# print("wancheng")
#
# for i in range(1):
#     print(i)