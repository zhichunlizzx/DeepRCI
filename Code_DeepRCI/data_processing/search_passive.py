import xlsxwriter

def search_pass(path):
    file_path = r'E:\data\passive.xlsx'
    workbook = xlsxwriter.Workbook(file_path)
    sheet = workbook.add_worksheet("passive_num")
    # 包含所有GO项，有冗余，用于计算个数
    go_anno_redundancy = []
    # 存储GO项和其对应的个数的关系
    go_anno_num = {}
    aaa = 0
    with open(path, mode='r') as r_pbj:
        raw_go_anno = r_pbj.readlines()
        r_pbj.close()
    for go_anno in raw_go_anno:
        per_go_list = go_anno.split('|')
        if len(per_go_list)==1:
            aaa += 1
        for per_go in per_go_list:
            go_anno_redundancy.append(per_go)
    go_anno_set = set(go_anno_redundancy)
    for go in go_anno_set:
        sum = 0
        for go_anno_re in go_anno_redundancy:
            if go == go_anno_re:
                sum += 1
        go_anno_num[go] = sum
    print(go_anno_num)
    rows = 0
    for key, value in go_anno_num.items():
        sheet.write(rows, 0, key)
        sheet.write(rows, 1, value)
        rows += 1
    print("The work is done")
    print(aaa)



if __name__ == '__main__':
    path = r'E:\data\GO_anno.txt'
    search_pass(path)