from Bio import SeqIO

def convert_a3m_to_aln(a3m_path, aln_path):
    aln = []
    with open(a3m_path, mode='r') as obj_a3m:
        for line in obj_a3m:
            if line[0] == '>':
                continue
            aln.append("".join([_ for _ in line if not _.islower()]))
    with open(aln_path, 'w') as obj_aln:
        obj_aln.writelines(aln)


def convert_a3m_to_aln_1000(a3m_path, aln_path):
    aln = []
    with open(a3m_path, mode='r') as obj_a3m:
        num = 0
        for line in obj_a3m:
            if line[0] == '>':
                continue
            aln.append("".join([_ for _ in line if not _.islower()]))
    with open(aln_path, 'w') as obj_aln:
        obj_aln.writelines(aln[:999])
        obj_aln.write(aln[999][:-1])


def num_of_seq(path):
    sum = 0
    for seq_data in SeqIO.parse(path, 'fasta'):
        sum += 1
    return sum

def per_protein_call(path, pre_path_a3m, pre_path_aln, id_path):
    id = []
    with open(path, mode='r') as r_obj:
        protein_ids = r_obj.readlines()
    # sum = 0
    for protein_id in protein_ids:
        num = 0
        a3m_path = pre_path_a3m + protein_id[:-1] + '.a3m'
        aln_path = pre_path_aln + protein_id[:-1] + '.aln'
        num = num_of_seq(a3m_path)
        if num >= 1000:
            convert_a3m_to_aln_1000(a3m_path, aln_path)
            id.append(protein_id)
            print(protein_id[:-1],'has down')
    with open(id_path, 'w') as w_obj:
        w_obj.writelines(id)
        w_obj.close()

    # print(sum)



if __name__ == '__main__':
    id_path = r'H:\data\passive\protein_id.txt'
    a3m_path_normal = r'H:\data\passive\a3m_passive\\'
    aln_path_normal = r'H:\data\passive\aln_passive\\'
    id_path_normal = r'H:\data\passive\aln_passive\protein_id.txt'

    id_path_max_E = r'H:\data\passive\less_aa.txt'
    a3m_path_max_E = r'H:\data\passive\a3m_passive_max_E\\'
    aln_path_max_E = r'H:\data\passive\aln_passive_max_E\\'
    id_path_max_E1 = r'H:\data\passive\aln_passive_max_E\protein_id.txt'

    # per_protein_call(id_path, a3m_path_normal, aln_path_normal, id_path_normal)
    # per_protein_call(id_path_max_E, a3m_path_max_E, aln_path_max_E, id_path_max_E1)
    # a3m_path = r'H:\data\0005524\a3m_0005524'
    # aln_path = r'H:\script\yigealn.aln'
    # convert_a3m_to_aln(a3m_path, aln_path)
    convert_a3m_to_aln(r'F:\Q9J0W9.a3m', r'F:\22.aln')