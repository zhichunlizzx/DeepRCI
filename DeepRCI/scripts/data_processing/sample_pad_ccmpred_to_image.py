from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from chest_failed_ccm import all_mat_name
import numpy as np
import matplotlib.pyplot as plt
import os

SHAPE = (400, 400)
CUTOFF = 10


def parse_protein_id(filename):
    #filename = "/tmp/d1a3aa_"
    #filename = "/tmp/d1a3aa_.fasta"
    name = filename.split('\\')[-1]
    protein_id = '.'.join(name.split('.')[:-1]) or name
    return protein_id


def sample_matrix(matrix):
    nums = np.random.choice(matrix.shape[0], SHAPE[0], replace=False)
    nums.sort()

    return matrix[np.ix_(nums, nums)] # equal to matrix[nums,:][:, nums]


def pad_matrix(matrix):
    new_matrix = np.zeros(SHAPE)
    start = np.random.randint(0, SHAPE[0] - matrix.shape[0])
    end = start + matrix.shape[0]
    new_matrix[start:end, start:end] = matrix
    
    return new_matrix


def sample_or_pad_matrix(matrix):
    # check data
    assert matrix.shape[0] == matrix.shape[1]

    # calculate random choose times
    num = abs(SHAPE[0] - matrix.shape[0])
    times = int(num // CUTOFF) + 1 # add pseudo-count 1

    # sample or pad matrix
    if matrix.shape[0] < SHAPE[0]:
        return [pad_matrix(matrix) for _ in range(times)]
    else:
        return [sample_matrix(matrix) for _ in range(times)]


def main(ccmpred_output, outdir):
    # parse name
    name = parse_protein_id(ccmpred_output)

    # load ccmpred outfile
    ccm_mat = np.loadtxt(ccmpred_output)

    # l = len(ccm_mat)
    #     # with open(r'F:\data_xmm\test_l_1.txt', 'a') as w_obj:
    #     #     w_obj.write(str(l))
    #     #     w_obj.write('\n')
    # return
    # for debug
    #np.random.seed(31415)
    # if len(ccm_mat) < 100:
    #     return
    # if len(ccm_mat) > 750:
    #     return
    # with open(r'F:\data_fan\train\1.txt', 'a') as w_obj:
    #     w_obj.write(ccmpred_output)
    #     w_obj.write('\n')

    # sample or pad matrix
    fix_mat = sample_or_pad_matrix(ccm_mat)
    assert isinstance(fix_mat, list)

    # check output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save file
    print('>%s' % name, ccm_mat.shape, SHAPE, len(fix_mat))
    for i, mat in enumerate(fix_mat):
        image_name = '%s/%s.%02d.jpg' % (outdir, name, i)
        print(image_name)
        plt.imsave(image_name, mat, cmap=plt.cm.gray_r)
        break




if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     sys.exit('Usage: %s <ccmpred_output> <outdir>' % sys.argv[0])
    # ccmpred_output, outdir = sys.argv[1:]
    file_path = r'F:\data\passive\mat_passive_normal'
    file_path_max_E = r'F:\data\passive\mat_passive'
    # mat_name = all_mat_name(file_path_max_E)
    # for mat in mat_name:
    #     ccmpred_output = file_path_max_E + '\\' + mat
    #     outdir = r'F:\data\train\0\\'
    #     print(ccmpred_output)
    #     main(ccmpred_output, outdir)
    #     print('{} has down'.format(mat.split('.')[0]))

    with open(r'F:\data_xmm\validation\0.txt', mode='r') as r_obj:
        mat_path = r_obj.readlines()
        # print(mat_path[0][:-1])
    out_path = r'F:\data_400_1\validation_1\0\\'
    for path in mat_path:
        path_mat = path[:-1]
        print('begin:', path_mat)
        main(path_mat, out_path)
        print('{} has down'.format(path_mat))

'''
    with open(r'F:\data\train\0\a.txt', mode='r') as r_obj:
        mat_path = r_obj.readlines()
        # print(mat_path[0][:-1])
    out_path = r'F:\data\train\0\\'
    for path in mat_path:
        path_mat = path[:-1]
        print('begin:',path_mat)
        main(path_mat, out_path)
        print('{} has down'.format(path_mat))
'''

'''
    with open(r'F:\data\validation\1\a.txt', mode='r') as r_obj:
        mat_path = r_obj.readlines()
        # print(mat_path[0][:-1])
    out_path = r'F:\data\validation\1\\'
    for path in mat_path:
        path_mat = path[:-1]
        print('begin:',path_mat)
        main(path_mat, out_path)
        print('{} has down'.format(path_mat))


    with open(r'F:\data\validation\0\a.txt', mode='r') as r_obj:
        mat_path = r_obj.readlines()
        # print(mat_path[0][:-1])
    out_path = r'F:\data\validation\0\\'
    for path in mat_path:
        path_mat = path[:-1]
        print('begin:',path_mat)
        main(path_mat, out_path)
        print('{} has down'.format(path_mat))
'''