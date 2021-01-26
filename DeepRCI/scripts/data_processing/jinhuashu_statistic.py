

def chuli_shu(path_tree, out_path):
    with open(path_tree, 'r') as r_obj:
        tree = r_obj.readlines()
    alphabet = ['p', 'n', 'p']
    new_tree = []
    num = 0
    for i in range(len(tree) - 1):
        if num > 0:
            num = num - 1
            continue
        if not ('(' in tree[i]) and (tree[i][0] in alphabet):
            if not ('(' in tree[i + 1]) and (tree[i + 1][0] in alphabet):
                pre = tree[i].split('|')
                later = tree[i + 1].split('|')
                pre, later = pre[0], later[0]
                if pre == later:
                    new_tree.pop()
                    new_tree.append(pre + '|' + tree[i + 2])
                    num = 2
                    continue
        new_tree.append(tree[i])
    new_tree.append(')')
    with open(out_path, 'w') as w_obj:
        for t in new_tree:
            w_obj.write(t)


def zhiliu_posi_nega_pre(path_tree, out_path):
    with open(path_tree, 'r') as r_obj:
        tree = r_obj.readlines()
    alphabet = ['p', 'n', 'p']
    new_tree = []
    num = 0
    for i in range(len(tree)):
        pre = tree[i].split('|')[0]
        later = tree[i].split(':')[-1]
        # print(pre)
        # print(tree[i][-8:])
        if pre == 'positive':
            new_tree.append('A' + ':' + later)
            continue
        if pre == 'negative':
            new_tree.append('N' + ':' + later)
            continue
        if pre == 'predicted':
            new_tree.append('P' + ':' + later)
            continue
        new_tree.append(tree[i])
    with open(out_path, 'w') as w_obj:
        for t in new_tree:
            w_obj.write(t)



def main():

    # path_tree = r'H:\sky\new_tree.txt'
    # path_out = r'H:\sky\new_tree.txt'
    # chuli_shu(path_tree, path_out)

    # path_tree = r'H:\sky\tree_1.txt'
    # path_out = r'H:\sky\new_tree_1.txt'
    # zhiliu_posi_nega_pre(path_tree, path_out)

    alphabet = ['p', 'n', 'p']
    path_tree = r'H:\sky\new_tree.txt'
    with open(path_tree, 'r') as r_obj:
        tree = r_obj.readlines()

    pos_pre = 0
    neg_pre = 0
    pos_neg = 0
    gap_pos_pre = 0
    gap_neg_pre = 0
    gap_pos_neg = 0
    for i in range(len(tree)-1):
        if not('(' in tree[i]) and (tree[i][0] in alphabet):
            if not('(' in tree[i+1]) and (tree[i+1][0] in alphabet) :
                pre = tree[i].split('|')
                later = tree[i+1].split('|')
                pre_value = float(tree[i].split(':')[-1][:-2])
                later_value = float(tree[i+1].split(':')[-1][:-2])



                pre, later = pre[0], later[0]
                if (pre == 'positive' and later == 'predicted') or (pre == 'predicted' and later == 'positive'):
                    pos_pre += 1
                    gap_pos_pre += abs(pre_value - later_value)
                elif (pre == 'negative' and later == 'predicted') or (pre == 'predicted' and later == 'negative'):
                    neg_pre += 1
                    gap_neg_pre += abs(pre_value - later_value)
                elif (pre == 'negative' and later == 'positive') or (pre == 'positive' and later == 'negative'):
                    pos_neg += 1
                    gap_pos_neg += abs(pre_value - later_value)
    print(pos_pre)
    print(neg_pre)
    print(pos_neg)
    #
    print(gap_pos_pre/pos_pre)
    print(gap_neg_pre/neg_pre)
    print(gap_pos_neg/pos_neg)





if __name__ == '__main__':
    main()