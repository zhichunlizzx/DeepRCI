



def main():
    path_seq = r'G:\之前电脑\H\data\seq0005524/'
    # path_seq = r'G:\之前电脑\H\data\seq_passive/'
    path_id = r'E:\my_research\DeepRCI\duli_ceshi\pos_test.txt'
    path_out = r'E:\my_research\DeepRCI\duli_ceshi\pos_test.fasta'

    with open(path_id, 'r') as r_obj:
        ids = r_obj.readlines()
    ids = [i[:-1] for i in ids]
    with open(path_out, 'w') as w_obj:
        for id in ids:
            with open(path_seq + id + '.fasta', 'r') as r_obj:
                fasta = r_obj.readlines()
            w_obj.writelines(fasta)



if __name__ == '__main__':
    main()