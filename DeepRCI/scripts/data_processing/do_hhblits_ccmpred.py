import os
import subprocess

def do_hhblits(path):
    # protein_id = []
    # d = subprocess.Popen('su')
    # subprocess.call('su')
    with open(path, mode='r') as r_obj:
        protein = r_obj.readlines()
    for pro in protein:
        # protein_id.append(pro)
        seq_id = pro[:-1]
        in_path = r'/home/zzx/data/0005524/seq0005524/' + seq_id + '.fasta'
        out_path = r'/home/zzx/data/0005524/a3m_0005524_max_E/' + seq_id + '.a3m'
        database_path = r'/home/zzx/hh-suite/data/pdb70_datasets/pdb70'
        # cmd = 'hhblits'+ ' ' + '-i' + ' ' + in_path + '-oa3m' +  ' ' + out_path + ' ' + '-d' + ' ' + database_path + '-n' + ' ' + '3'
        cmd = ['hhblits', '-i', in_path, '-oa3m', out_path, '-d', database_path, '-n', '3', '-e', '1000']
        print(cmd)
        # os.system(cmd)
        subprocess.call(cmd)
        print(pro + ' ' + 'has down')


def do_Pseaac(path):
    with open(path, mode='r') as r_obj:
        protein = r_obj.readlines()
        for pro in protein:
            seq_id = pro[:-1]
            in_path = r'/home/ubuntu/xibaoyinzi/positive/per_seq/' + seq_id + '.fasta'
            out_path = r'/home/ubuntu/xibaoyinzi/positive/pseaac_pos/' + seq_id + '.csv'
            cmd = ['pseb/pseb', '-t', '0', '-i', in_path, '-o', out_path, '-m', 'csv', '-l', '10', '-w', '0.05', '-a']
            print(cmd)
            subprocess.call(cmd)
            print(pro + ' ' + 'has down')


def a3m_to_aln(a3m_path, aln_path):
    aln = []
    with open(a3m_path, mode='r') as obj_a3m:
        for line in obj_a3m:
            if line[0] == '>':
                continue
            aln.append("".join([_ for _ in line if not _.islower()]))
    with open(aln_path, 'w') as obj_aln:
        obj_aln.writelines(aln)


def do_ccmpred(path, aln_path_pre, out_path_pre, failed_path):
    # fail_mat_id =[]
    with open(path, mode='r') as r_obj:
        protein = r_obj.readlines()
    for protein_id in protein:
        seq_id = protein_id[:-1]
        #未完待续
        # a3m_path = r'/home/zzx/data_xi/a3m_passive/' + seq_id + '.a3m'
        aln_path = aln_path_pre + seq_id + '.aln'
        out_path = out_path_pre + seq_id + '.mat'
        # a3m_to_aln(a3m_path, aln_path)
        cmd = ['ccmpred', aln_path, out_path, '-n', '50']
        subprocess.call(cmd)
        if os.path.exists(out_path):
            print(seq_id[:-1] + ' ' + 'has down')
        else:
            # fail_mat_id.append(protein_id)
            with open(failed_path, mode='a') as w_obj:
                w_obj.write(protein_id)
                w_obj.close()


def make_pssm(path, input_path_pre, output_path_pre_pssm, output_path_pre):
    with open(path, mode='r') as r_obj:
        protein = r_obj.readlines()
    for protein_id in protein:
        seq_id = protein_id[:-1]
        seq_id = seq_id.split('\\')
        seq_id = seq_id[-1]
        seq_id = seq_id[:-4]
        input_path = input_path_pre + seq_id + '.fasta'
        output_path_pssm = output_path_pre_pssm + seq_id + '.pssm'
        output_path = output_path_pre + seq_id + '.txt'
        cmd = ['psiblast', '-evalue', '1000', '-num_iterations','3', '-db', './zzx/ncbi-blast-2.10.0+/db/swissprot.fasta', '-query', input_path, '-out', output_path, '-out_ascii_pssm', output_path_pssm]
        subprocess.call(cmd)
        print(seq_id + ' ' + 'has down')

def make_SSS(path, input_path_pre, output_path_pre):
    with open(path, mode='r') as r_obj:
        protein = r_obj.readlines()
    for protein_id in protein:
        seq_id = protein_id[:-1]
        seq_id = seq_id.split('\\')
        seq_id = seq_id[-1]
        seq_id = seq_id[:-4]
        input_path = input_path_pre + seq_id + '.fasta'
        # output_path_pssm = output_path_pre_pssm + seq_id + '.pssm'
        output_path = output_path_pre + seq_id + '.txt'
        cmd = ['./runpsipredplus',input_path, output_path]
        subprocess.call(cmd)
        print(seq_id + ' ' + 'has down')



if __name__ == '__main__':
    # protein_id_path = r'/home/zzx/data/0005524/less_aa.txt'
    # normal a3m
    # ccmpred_protein_path_normal = r'/home/zzx/data_xi/0005524/aln_0005524/protein_id8.txt'
    # aln_path_normal = r'/home/zzx/data_xi/0005524/aln_0005524/'
    # out_path_normal = r'/home/zzx/data_xi/0005524/mat_0005524_8/'
    # failed_path_normal =r'/home/zzx/data_xi/0005524/failed_normal.txt'
    # # do_ccmpred(ccmpred_protein_path_normal, aln_path_normal, out_path_normal, failed_path_normal)
    #
    # # max_E_value_a3m
    # # ccmpred_protein_path_max_E = r'home/zzx/data_xi/0005524/aln_0005524_max_E/protein_id.txt'
    # ccmpred_protein_path_max_E = r'/home/zzx/data_xi/0005524/aln_0005524_max_E/protein_id.txt'
    # aln_path_max_E = r'/home/zzx/data_xi/0005524/aln_0005524_max_E/'
    # out_path_max_E = r'/home/zzx/data_xi/0005524/mat_0005524_max_E'
    # failed_path_max_E = r'/home/zzx/data_xi/0005524/failed_max_E.txt'
    # # do_ccmpred(ccmpred_protein_path_max_E, aln_path_max_E, out_path_max_E, failed_path_max_E)
    # # do_hhblits(protein_id_path)
    # # do_ccmpred(ccmpred_protein_path_normal)
    # path = r'/home/ubuntu/1.txt'
    # seq_path = r'/home/ubuntu/0005524_seq/'
    # pssm_out = r'/home/ubuntu/passive_pssm/'
    # out = r'/home/ubuntu/1/'
    # # make_pssm(path, seq_path,pssm_out, out)
    # make_SSS(path, seq_path, out)
    path = r'/home/ubuntu/xibaoyinzi/positive/protein_id.txt'
    do_Pseaac(path)


    pass
