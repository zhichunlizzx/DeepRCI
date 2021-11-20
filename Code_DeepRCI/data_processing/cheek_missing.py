from Bio import SeqIO


def cheek():
    fasta_id = []
    miss = []
    file_0005524 = r'H:\0005524.txt'
    file_uniprot_code = r'E:\data\uniprot_code.txt'
    for fasta in SeqIO.parse(file_0005524, 'fasta'):
        fasta_id.append(fasta.id)
    with open(file_uniprot_code, mode='r') as code_obj:
        code = code_obj.readlines()
        code_obj.close()
    # 本来可以设置找完这个下个在找的时候可以从这个的位置开始
    # 记录缺失数
    sum_miss = 0
    for cod in code:
        sum = 0
        # row = 0
        # print(cod)
        for fasta_code in fasta_id:
            # print(fasta_code)
            if cod[10:-1] in fasta_code:
                # row = fasta_id.index(fasta_code)
                sum = 1
                break
        if sum == 0:
            sum_miss += 1
            print(cod)
            miss.append(cod)
        print(sum_miss)
    print("it's missing altogether:", sum_miss)
    with open(r'E:\data\miss1.txt', mode='a') as w_obj:
        for mis in miss:
            w_obj.write(mis)
        w_obj.close()


if __name__ == '__main__':
    cheek()
    pass