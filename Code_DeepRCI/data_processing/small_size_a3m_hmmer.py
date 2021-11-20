import os


def find_a3m_1kb(path):
    with open(path, mode='r') as r_obj:
        protein_ids = r_obj.readlines()
    sum = 0
    for protein_id in protein_ids:
        pro_path = r'H:\data\passive\a3m_passive\\' + protein_id[:-1] + '.a3m'
        # sum += 1
        size = os.path.getsize(pro_path)
        if size < 100000:
            sum += 1

        print(size)
    print('sum = ',sum)


if __name__ == '__main__':
    path = r'H:\data\passive\seq_passive\protein_id.txt'
    find_a3m_1kb(path)