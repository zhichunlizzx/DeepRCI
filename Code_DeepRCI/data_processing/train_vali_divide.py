import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

SHAPE = (256, 256)
CUTOFF = 10


def parse_protein_id(filename):
    # filename = "/tmp/d1a3aa_"
    # filename = "/tmp/d1a3aa_.fasta"
    name = filename.split('\\')[-1]
    protein_id = '.'.join(name.split('.')[:-1]) or name
    return protein_id


def sample_matrix(matrix):
    new_matrix = np.zeros(SHAPE)
    nums = np.random.choice(matrix.shape[0], SHAPE[0], replace=False)
    nums.sort()

    return matrix[np.ix_(nums, nums)]  # equal to matrix[nums,:][:, nums]


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
    times = int(num // CUTOFF) + 1  # add pseudo-count 1

    # sample or pad matrix
    if matrix.shape[0] < SHAPE[0]:
        return [pad_matrix(matrix) for _ in range(times)]
    else:
        return [sample_matrix(matrix) for _ in range(times)]

def load_mat_file_path(path_normal, path_max_E):
    dir_list_normal = os.listdir(path_normal)
    dir_list_max_E = os.listdir(path_max_E)
    path_list_normal = []
    path_list_max_E = []
    for dir in dir_list_normal:
        path = path_normal+ '\\'+ dir
        print(path)
        path_list_normal.append(path)
    for dir in dir_list_max_E:
        path = path_max_E + '\\' + dir
        path_list_max_E.append(path)

    path_list = np.append(path_list_normal, path_list_max_E)
    return path_list
    pass

def make_train_val_path():
    file_path_1_normal = r'F:\data\0005524\mat_0005524'
    file_path_1_max_E = r'F:\data\0005524\mat_0005524_max_E'
    mat_file_path_list_1 = load_mat_file_path(file_path_1_normal, file_path_1_max_E)
    # print(mat_file_path_list_1.shape)

    file_path_0_normal = r'F:\data\passive\mat_passive_normal'
    file_path_0_max_E = r'F:\data\passive\mat_passive'
    mat_file_path_list_0 = load_mat_file_path(file_path_0_normal,file_path_0_max_E)
    # print(mat_file_path_list_0.shape)
    # print(mat_file_path_list_1[0])
    idx_1 = tf.range(5170)
    idx_1 = tf.random.shuffle(idx_1)
    train_1, test_1 = tf.gather(mat_file_path_list_1, idx_1[:4500]), tf.gather(mat_file_path_list_1, idx_1[4500:])
    # print(train_1[0])
    train_1, val_1 = train_1[:4000], train_1[4000:]
    train_1, val_1 = np.asarray(train_1), np.asarray(val_1)
    test_1 = np.asarray(test_1)
    # print(train_1.shape)
    # print(train_1[0])
    idx_0 = tf.range(5315)
    idx_0 = tf.random.shuffle(idx_0)
    train_0, test_0 = tf.gather(mat_file_path_list_0, idx_0[:4500]), tf.gather(mat_file_path_list_0, idx_0[4500:])
    train_0, val_0 = train_0[:4000], train_0[4000:]
    train_0, val_0 = np.asarray(train_0), np.asarray(val_0)
    test_0 = np.asarray(test_0)
    return train_1, val_1,test_1, train_0, val_0, test_0
    pass


def main(ccmpred_output, outdir):
    # parse name
    name = parse_protein_id(ccmpred_output)

    # load ccmpred outfile
    ccm_mat = np.loadtxt(ccmpred_output)

    # for debug
    #np.random.seed(31415)

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

if __name__ == '__main__':
    train_1, val_1, test_1, train_0, val_0, test_0 = make_train_val_path()
    train_1_out = r'F:\data\train\1\\'
    val_1_out = r'F:\data\validation\1\\'
    train_0_out = r'F:\data\train\0\\'
    val_0_out = r'F:\data\validation\0\\'
    path_train_1 = r'F:\data_xmm\train\1.txt'
    path_train_0 = r'F:\data_xmm\train\0.txt'
    path_val_1 = r'F:\data_xmm\validation\1.txt'
    path_val_0 = r'F:\data_xmm\validation\0.txt'
    path_test_1 = r'F:\data_xmm\test\1.txt'
    path_test_0 = r'F:\data_xmm\test\0.txt'



    for mat_path in train_1:
        # print('byte:',mat_path)
        path = str(mat_path)
        # print('str:',path)
        del_list = []
        del1 = 0
        for i in range(len(path)):
            if del1 == 1:
                del_list.append(i)
                del1 = 0
                continue
            if path[i] == '\\'[0]:
                    del1 = 1
        # print(del_list)
        path1 = path
        path = list(path)
        pppath = ''
        for i in del_list[::-1]:
            path.pop(i)
        for i in path:
            pppath += i
        pppath = pppath[2:-1]
        # print(pppath)
        # path = path.replace('\\', '\\')
        with open(path_train_1, mode='a') as w_obj:
            a = pppath + '\n'
            path1 = path1 + '\n'
            # w_obj.write(path1)
            w_obj.write(a)
        # main(pppath, train_1_out)
        # print('{} has down'.format(mat_path))


    for mat_path in val_1:
        # print('byte:',mat_path)
        path = str(mat_path)
        # print('str:',path)
        del_list = []
        del1 = 0
        for i in range(len(path)):
            if del1 == 1:
                del_list.append(i)
                del1 = 0
                continue
            if path[i] == '\\'[0]:
                    del1 = 1
        # print(del_list)
        path = list(path)
        pppath = ''
        for i in del_list[::-1]:
            path.pop(i)
        for i in path:
            pppath += i
        pppath = pppath[2:-1]
        # print(pppath)
        # path = path.replace('\\', '\\')
        with open(path_val_1, mode='a') as w_obj:
            a = pppath + '\n'
            w_obj.write(a)
        # main(pppath, val_1_out)
        # print('{} has down'.format(mat_path))


    for mat_path in train_0:
        # print('byte:',mat_path)
        path = str(mat_path)
        # print('str:',path)
        del_list = []
        del1 = 0
        for i in range(len(path)):
            if del1 == 1:
                del_list.append(i)
                del1 = 0
                continue
            if path[i] == '\\'[0]:
                    del1 = 1
        # print(del_list)
        path = list(path)
        pppath = ''
        for i in del_list[::-1]:
            path.pop(i)
        for i in path:
            pppath += i
        pppath = pppath[2:-1]
        # print(pppath)
        # path = path.replace('\\', '\\')
        with open(path_train_0, mode='a') as w_obj:
            a = pppath + '\n'
            w_obj.write(a)
        # main(pppath, train_0_out)
        # print('{} has down'.format(mat_path))

    for mat_path in val_0:
        # print('byte:',mat_path)
        path = str(mat_path)
        # print('str:',path)
        del_list = []
        del1 = 0
        for i in range(len(path)):
            if del1 == 1:
                del_list.append(i)
                del1 = 0
                continue
            if path[i] == '\\'[0]:
                    del1 = 1
        # print(del_list)
        path = list(path)
        pppath = ''
        for i in del_list[::-1]:
            path.pop(i)
        for i in path:
            pppath += i
        pppath = pppath[2:-1]
        # print(pppath)
        # path = path.replace('\\', '\\')
        with open(path_val_0, mode='a') as w_obj:
            a = pppath + '\n'
            w_obj.write(a)
        # main(pppath, val_0_out)
        # print('{} has down'.format(mat_path))

    for mat_path in test_1:
        # print('byte:',mat_path)
        path = str(mat_path)
        # print('str:',path)
        del_list = []
        del1 = 0
        for i in range(len(path)):
            if del1 == 1:
                del_list.append(i)
                del1 = 0
                continue
            if path[i] == '\\'[0]:
                    del1 = 1
        # print(del_list)
        path1 = path
        path = list(path)
        pppath = ''
        for i in del_list[::-1]:
            path.pop(i)
        for i in path:
            pppath += i
        pppath = pppath[2:-1]
        # print(pppath)
        # path = path.replace('\\', '\\')
        with open(path_test_1, mode='a') as w_obj:
            a = pppath + '\n'
            path1 = path1 + '\n'
            # w_obj.write(path1)
            w_obj.write(a)


    for mat_path in test_0:
        # print('byte:',mat_path)
        path = str(mat_path)
        # print('str:',path)
        del_list = []
        del1 = 0
        for i in range(len(path)):
            if del1 == 1:
                del_list.append(i)
                del1 = 0
                continue
            if path[i] == '\\'[0]:
                    del1 = 1
        # print(del_list)
        path1 = path
        path = list(path)
        pppath = ''
        for i in del_list[::-1]:
            path.pop(i)
        for i in path:
            pppath += i
        pppath = pppath[2:-1]
        # print(pppath)
        # path = path.replace('\\', '\\')
        with open(path_test_0, mode='a') as w_obj:
            a = pppath + '\n'
            path1 = path1 + '\n'
            # w_obj.write(path1)
            w_obj.write(a)