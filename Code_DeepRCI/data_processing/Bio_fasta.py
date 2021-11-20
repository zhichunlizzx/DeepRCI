from Bio import SeqIO



def ex_seq_from_fasta(path):
    # fasta = {}
    fasta = []
    sum = 0
    print('1111111111')
    for seq_data in SeqIO.parse(path, 'fasta'):
        print('1')
        print(seq_data.id)
        print(seq_data.seq)
        # fasta.append(seq_data.seq)
        # fasta_1.append(seq_data.id)
        # print(seq_data.name)
       # sum = sum+1
    # for id in fasta:
    #     sum1 = 0
    #     print(id)
    #     for id1 in fasta_1:
    #         if id == id1:
    #             sum1 = sum1+1

        # print(sum1)


    # return fasta

def length_statistic(path):
    sum = 0
    for seq_data in SeqIO.parse(path, 'fasta'):
        if len(seq_data.seq) < 20:
            sum += 1
    return sum

def num_of_seq(path):
    sum = 0
    for seq_data in SeqIO.parse(path, 'fasta'):
        sum += 1
    return sum


def checking_num_msa(path):
    sum = 0
    with open(path, mode='r') as r_obj:
        aa_ids = r_obj.readlines()
    for aa_id in aa_ids:
        aa_path = r'H:\data\passive\a3m_passive\\' + aa_id[:-1] + '.a3m'
        w_path = r'H:\data\passive\less_aa.txt'
        # if aa_id == aa_ids[-1]:
        #     aa_path = r'H:\data\0005524\a3m_0005524\\' + aa_id + '.a3m'
        # print(aa_path)
        num = num_of_seq(aa_path)
        if num < 1000:
            sum += 1
            print(aa_id)
            print(num)
            with open(w_path, mode='a') as w_obj:
                w_obj.write(aa_id)
                w_obj.close()
    print(sum)


if __name__ == '__main__':
    # path = r'E:\data\A0A0A0K6G7.fasta'
    # path = r'H:\data\0005524\protein_id.txt'
    path = r'C:\dataz\research\DeepRCI\duli_ceshi\atp_train.fasta'
    # checking_num_msa(path)
    # print(ex_seq_from_fasta(path))
    # print(length_statistic(path))
    print(num_of_seq(path))
    # ex_seq_from_fasta(path)
