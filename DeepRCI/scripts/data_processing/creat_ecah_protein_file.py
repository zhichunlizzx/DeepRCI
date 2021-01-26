# coding:utf-8

from Bio import SeqIO
import os

def creat_file(source_file):
    for data in SeqIO.parse(source_file, 'fasta'):
        acid_id = data.id
        # if len(data.seq) <= 800:
            # 把蛋白质名词取出来，然后当成文件名
            # 将data.id和data.seq写入
        num = 0
        num_start = 0
        num_end = 0
        for i in range(len(acid_id)):
            if (acid_id[i] == '|')&(num == 0):
                num_start = i
                num += 1
            if(acid_id[i] == '|')&(num != 0):
                num_end = i
                continue
        print(acid_id[num_start+1:num_end]+'.fasta is creat')
        with open(r'H:\xibaoyinzi\data\positive/protein_id.txt',mode='a') as w_obj:
            w_obj.write(acid_id[num_start+1:num_end]+'\n')
            w_obj.close()
        seq_file = r'H:\xibaoyinzi\data\positive\per_seq\\' + acid_id[num_start+1:num_end] + '.fasta'
        SeqIO.write(data, seq_file, 'fasta')
            # with open(seq_file,'a') as w_obj:
            #     w_obj.write(data.id)
            #     w_obj.write(data.seq)


        # print(data)
        # print('---')
        pass




if __name__ == '__main__':
    creat_file(r'H:\xibaoyinzi\data\positive\seq\seq.txt')
    pass
