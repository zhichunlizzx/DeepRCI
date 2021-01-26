from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt

def AAC(path, file_path):
    with open(path, 'r') as r_obj:
        p_id = r_obj.readlines()
    p_id = [i[:-1] for i in p_id]
    # AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
    #         'L', 'M', 'N', 'R', 'P', 'Q', 'S', 'T',
    #         'V', 'W', 'Y']
    a = 0
    c = 0
    d = 0
    e = 0
    f = 0
    g = 0
    h = 0
    i = 0
    k = 0
    l = 0
    m = 0
    n = 0
    p = 0
    q = 0
    r = 0
    s = 0
    t = 0
    v = 0
    w = 0
    y = 0
    ll = 0
    for id in p_id:
        for pro in SeqIO.parse(file_path + id + '.fasta', 'fasta'):
            seq = pro.seq
            a += seq.count('A')
            c += seq.count('C')
            d += seq.count('D')
            e += seq.count('E')
            f += seq.count('F')
            g += seq.count('G')
            h += seq.count('H')
            i += seq.count('I')
            k += seq.count('K')
            l += seq.count('L')
            m += seq.count('M')
            n += seq.count('N')
            r += seq.count('R')
            p += seq.count('P')
            q += seq.count('Q')
            s += seq.count('S')
            t += seq.count('T')
            v += seq.count('V')
            w += seq.count('W')
            y += seq.count('Y')
            ll += len(seq)

    aa_c = []
    aa_c.append(a / ll)
    aa_c.append(c / ll)
    aa_c.append(d / ll)
    aa_c.append(e / ll)
    aa_c.append(f / ll)
    aa_c.append(g / ll)
    aa_c.append(h / ll)
    aa_c.append(i / ll)
    aa_c.append(k / ll)
    aa_c.append(l / ll)
    aa_c.append(m / ll)
    aa_c.append(n / ll)
    aa_c.append(r / ll)
    aa_c.append(p / ll)
    aa_c.append(q / ll)
    aa_c.append(s / ll)
    aa_c.append(t / ll)
    aa_c.append(v / ll)
    aa_c.append(w / ll)
    aa_c.append(y / ll)
    return aa_c


def AAC_fasta(path):
    # with open(path, 'r') as r_obj:
    #     p_id = r_obj.readlines()
    # p_id = [i[:-1] for i in p_id]
    # AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
    #         'L', 'M', 'N', 'R', 'P', 'Q', 'S', 'T',
    #         'V', 'W', 'Y']
    a = 0
    c = 0
    d = 0
    e = 0
    f = 0
    g = 0
    h = 0
    i = 0
    k = 0
    l = 0
    m = 0
    n = 0
    p = 0
    q = 0
    r = 0
    s = 0
    t = 0
    v = 0
    w = 0
    y = 0
    ll = 0
    # for id in p_id:
    for pro in SeqIO.parse(path, 'fasta'):
        seq = pro.seq
        a += seq.count('A')
        c += seq.count('C')
        d += seq.count('D')
        e += seq.count('E')
        f += seq.count('F')
        g += seq.count('G')
        h += seq.count('H')
        i += seq.count('I')
        k += seq.count('K')
        l += seq.count('L')
        m += seq.count('M')
        n += seq.count('N')
        r += seq.count('R')
        p += seq.count('P')
        q += seq.count('Q')
        s += seq.count('S')
        t += seq.count('T')
        v += seq.count('V')
        w += seq.count('W')
        y += seq.count('Y')
        ll += len(seq)

    aa_c = []
    aa_c.append(a / ll)
    aa_c.append(c / ll)
    aa_c.append(d / ll)
    aa_c.append(e / ll)
    aa_c.append(f / ll)
    aa_c.append(g / ll)
    aa_c.append(h / ll)
    aa_c.append(i / ll)
    aa_c.append(k / ll)
    aa_c.append(l / ll)
    aa_c.append(m / ll)
    aa_c.append(n / ll)
    aa_c.append(r / ll)
    aa_c.append(p / ll)
    aa_c.append(q / ll)
    aa_c.append(s / ll)
    aa_c.append(t / ll)
    aa_c.append(v / ll)
    aa_c.append(w / ll)
    aa_c.append(y / ll)
    return aa_c


def huatu(aac1, aac2, aac3):
    n_groups = 20

    # means_men = (0.9829, 0.9881, 0.978, 0.9657)
    # std_men = (1, 1, 1, 1, 1)

    # means_women = (0.89, 0.87, 0.92, 0.79)
    # std_women = (3, 5, 2, 3, 3)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.2

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, aac1, bar_width,
                    alpha=opacity, color='b',
                    error_kw=error_config,
                    label='Negative example')

    rects2 = ax.bar(index + bar_width, aac2, bar_width,
                    alpha=opacity, color='r',
                    error_kw=error_config,
                    label="ATP-Binding protein")

    rects3 = ax.bar(index + 2 * bar_width, aac3, bar_width,
                    alpha=opacity, color='y',
                    error_kw=error_config,
                    label="predicted ATP-Binding protein")

    # rects4 = ax.bar(index + 3 * bar_width, aac4, bar_width,
    #                 alpha=opacity, color='g',
    #                 error_kw=error_config,
    #                 label="True negatives")
    # for x1, yy in enumerate(aac1):
    #     plt.text(x1, yy, str(yy), ha='center', va='bottom', fontsize=10, rotation=0)
    # for x1, yy in enumerate(aac2):
    #     plt.text(x1 + bar_width, yy, str(yy), ha='center', va='bottom', fontsize=10, rotation=0)
    # ax.set_xlabel('Group')
    # ax.set_ylabel('Scores')
    plt.ylim([0, 0.15])
    ax.set_title("Amino acid composition")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
            'L', 'M', 'N', 'R', 'P', 'Q', 'S', 'T',
            'V', 'W', 'Y'))
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    file_path_1 = r'G:\之前电脑\H\data\seq_passive/'
    path_1 = r'G:\之前电脑\H\data\seq_passive\protein_id.txt'
    aac_1 = AAC(path_1, file_path_1)
    file_path_2 = r'G:\之前电脑\H\data\seq0005524/'
    path_2 = r'G:\之前电脑\H\data\seq0005524\protein_id.txt'
    aac_2 = AAC(path_2, file_path_2)
    # file_path_3 = r'E:\my_research\transporter_svm\data_jiaozheng\neg_9k/'
    path_3 = r'H:\sky\ATP-BPs_predicited.txt'
    aac_3 = AAC_fasta(path_3)
    # file_path_4 = r'E:\my_research\transporter_svm\data_jiaozheng\neg_9k/'
    # path_4 = r'E:\my_research\transporter_svm\data_jiaozheng\neg_id.txt'
    # aac_4 = AAC(path_4, file_path_4)


    # print(aac_1)
    # print(aac_2)
    huatu(aac_2, aac_1, aac_3)