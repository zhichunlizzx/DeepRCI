import numpy as np
import random

def main():

    path_abc = r'E:\my_research\DeepRCI\duli_ceshi\abc.txt'

    path_atp = r'E:\my_research\DeepRCI\duli_ceshi\atp.txt'

    with open(path_abc, 'r') as r_obj:
        abc_ids = r_obj.readlines()
    with open(path_atp, 'r') as r_obj:
        atp_ids = r_obj.readlines()

    abc_duli = []
    for id in abc_ids:
        if not(id in atp_ids):
            abc_duli.append(id)
        else:
            print('111111')
    # print(abc_duli)
    random.shuffle(abc_duli)
    # print(abc_duli)

    # with open(r'E:\my_research\DeepRCI\duli_ceshi\abc_quchong.txt', 'w') as w_obj:
    #     w_obj.writelines(abc_duli)


if __name__ == '__main__':
    main()