import os
import numpy as np

def chest_ccm(mat_name, path):
    failed_path = r'F:\data\passive\failed_ccm_passive.txt'
    failed_ccm = []
    for mat in mat_name:
        protein_id = mat.split('.')[0] + '\n'
        print(protein_id)
        try:
            ccm = np.loadtxt(path + '\\' + mat)
        except Exception:
            failed_ccm.append(protein_id)
            continue

        # ccm = np.loadtxt(path + '\\' + mat)
        print(ccm.shape)
        print(type(ccm))
        # print(protein_id)
        # print('*' * 50)


        # if len(ccm) == 0:
        #     # protein_id = mat.split('.')[0]
        #     failed_ccm.append(protein_id)
        try:
            if ccm.shape[0] != ccm.shape[1]:
                # protein_id = mat.split('.')[0]
                failed_ccm.append(protein_id)
        except IndexError:
            failed_ccm.append(protein_id)
        print('*' * 50)
        # if (ccm.shape[0] != ccm.shape[1]) or (len(ccm) == 0):
        #     protein_id = mat.split('.')[0]
        #     failed_ccm.append(protein_id)
    with open(failed_path, mode='w') as w_obj:
        w_obj.writelines(failed_ccm)
    print("-"*50)

def all_mat_name(path):
    mat_name = os.listdir(path)
    return mat_name

def main():
    path = r'F:\data\passive\mat_passive_normal'
    mat_name = all_mat_name(path)
    chest_ccm(mat_name, path)


if __name__ == '__main__':
    main()