import xlrd
from Bio import SeqIO
import os
import tensorflow as tf
import numpy as np

def search_protein_id(path):
    try:
        workbook = xlrd.open_workbook(path)
    except Exception as e:
        print(str(e))
    table = workbook.sheet_by_name('Integrated_Function.anno')
    total_rows = table.nrows
    # columns = table.row_values(0)
    protein_id = []
    for i in range(1,total_rows):
        columns = table.row_values(i)
        if columns[3] == '--':
            protein_id.append(columns[0])
    with open(r'E:\qizhi_data\unlabeled_protein_id.txt','w') as w_obj:
        for id in protein_id:
            w_obj.write(id)
            w_obj.write('\n')


def search_sequence(path):
    path_sequence = r'E:\qizhi_data\Glycine_soja.Unigene.pep.fa'
    path_out_file = r'E:\qizhi_data\unlabeled800\\'
    path_id = r'E:\qizhi_data\id_800.txt'
    with open(path, 'r') as r_obj:
        id = r_obj.readlines()
    for seq in SeqIO.parse(path_sequence, 'fasta'):
        if len(seq.seq)<= 800:
            for id_1 in id:
                id_1 = id_1[:-1]
                if id_1 in seq.id:
                    with open(path_id, 'a') as w_obj:
                        w_obj.write(id_1)
                        w_obj.write('\n')
                    path_out = path_out_file + id_1 + '.fasta'
                    SeqIO.write(seq, path_out, 'fasta')


def search_out_uniprot(path):
    path_sequence = r'E:\qizhi_data\unlabeled800\\'
    protein_dir = os.listdir(path_sequence)
    uniprot = []
    out_uni = []
    for uni in SeqIO.parse(r'E:\data\uniprot_sprot.fasta', 'fasta'):
        uniprot.append(uni)
    for dir in protein_dir:
        dir = path_sequence + dir
        for seq in SeqIO.parse(dir, 'fasta'):
            for uni in uniprot:
                if seq.seq == uni.seq:
                    out_uni.append(dir)
                    print(uni.id)
    for u in out_uni:
        with open(r'E:\qizhi_data\uniprot_sprot.fasta', 'a') as w_obj:
            w_obj.write(u)
            w_obj.write('\n')


def random_select():
    path = r'E:\qizhi_data\unlabeled800/'
    path_out = r'E:\qizhi_data\unlabeled_2000/'
    protein_file = os.listdir(path)
    idx = tf.range(23311)
    idx = tf.random.shuffle(idx)
    protein_sele = tf.gather(protein_file, idx[:2000])
    protein_sele = np.asarray(protein_sele)
    for protein in protein_sele:
        protein = str(protein)[2:-1]
        protein_in = path + str(protein)
        protein_out = path_out + protein
        for i in SeqIO.parse(protein_in, 'fasta'):
            SeqIO.write(i, protein_out, 'fasta')


def main():
    path = r'E:\qizhi_data\unlabeled_protein_id.txt'
    # search_protein_id(path)
    search_out_uniprot(path)

if __name__ == '__main__':
    # main()
    random_select()