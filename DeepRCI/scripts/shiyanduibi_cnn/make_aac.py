from Bio import SeqIO


def one_gram(path):
    for pro in SeqIO.parse(path, 'fasta'):
        seq = pro.seq
    p_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
            'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'V', 'W', 'Y']

    one_gram = []
    for aa in p_aa:
        one_gram.append(seq.count(aa) / len(seq))

    return one_gram


def s_gram(path):
    for data in SeqIO.parse(path, 'fasta'):
        seq = data.seq
    seq = str(seq)

    p_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
              'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
              'V', 'W', 'Y']
    s_gram_400 = []

    for aa_i in p_aa:
        for aa_j in p_aa:
            aa_s = aa_i+aa_j
            s_gram_400.append(aa_s)
    s_gram = []
    for t in s_gram_400:
        num = seq.count(t)
        num = num / (len(seq) - 1)
        s_gram.append(num)
    return s_gram


def t_gram_normal(path):

    for data in SeqIO.parse(path, 'fasta'):
        seq = data.seq
    seq = str(seq)

    p_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
            'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'V', 'W', 'Y']
    t_gram_512 = []

    for  aa_i in p_aa:
        for aa_j in p_aa:
            for aa_k in p_aa:
                aa_t = aa_i + aa_j + aa_k
                t_gram_512.append(aa_t)
    # print(len(t_gram_512))
    t_gram = []
    for t in t_gram_512:
        num = seq.count(t)
        num = num / (len(seq) - 2)
        t_gram.append(num)
    return t_gram

def main():
    path = r'G:\之前电脑\H\data\seq_passive/'
    id_path = r'E:\data\huafen\val\0.txt'
    out_path = r'E:\data\AAC\one_gram\t_gram_val.csv'
    label = '0'
    with open(id_path , 'r') as r_obj:
        ids = r_obj.readlines()
    ids = [i[:-1] for i in ids]
    # with open(out_path, 'a') as w_obj:
    #     w_obj.write('feature' + ',' + 'label' + '\n')

    for id in ids:
        path_id = path + id + '.fasta'
        aac = t_gram_normal(path_id)
        with open(out_path, 'a') as w_obj:
            for value in aac:
                w_obj.write(str(value) + ',')
            w_obj.write(label)
            w_obj.write('\n')


if __name__ == '__main__':
    main()