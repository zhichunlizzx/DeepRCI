import xlrd
import xlwt

file_path = r'E:\data\All_Database_annotation.xls'

def open_excel():
    """
    读取excel数据的索引
    :return: excel数据的一个索引
    """
    data = xlrd.open_workbook(file_path)
    #print(data)
    return data

def extract_info():
    data = open_excel()
    table = data.sheet_by_name('GO.list')
    #读取excel总行数
    rows = table.nrows
    # print(rows)
    #print(list(range(rows)))
    #excel也是按从第0行开始的
    #function_statistic
    fun_sta = {}
    for row_num in range(rows):
        #每行的某列是否有值
        row_bool = table.row_types(row_num)
        #print(row_data)
        #获取每行的数据
        row_value = table.row_values(row_num)
        #print(row_value)
        key = row_value[0]
        #fun_sta[key] =
        # 记录每一行的功能编号
        k_value = []
        for i in range(1,len(row_bool)):

            if row_bool[i] == 1:
                k_value.append(row_value[i])
        fun_sta[key] = k_value
    return(fun_sta)

def func_num_list():
    fun_sta = extract_info()
    #建立一个列表存储全部功能编号
    raw_func_num = []
    for key,value in fun_sta.items():
        for va in value:
            raw_func_num.append(va)
    return (raw_func_num)

def func_static():
    raw_func = func_num_list()
    #将列表转换成集合，再转换成列表就可以实现去重
    f = set(raw_func)
    func_list = list(f)

    # print(len(raw_func))
    # print(len(func_list))
    # print(func_list)
    func_num = {}
    for key in func_list:
        num = 0
        for func in raw_func:
            if func == key:
                num = num + 1
        func_num[key] = num
    return func_num

def w_to_exc(func_static):
    """
    将统计结果写入excel
    :return: None
    """
    f_path = r'E:\data\result.xls'
    workbook = xlwt.Workbook()
    #func_num_statistic = workbook.add_sheet('func_num_static',cell_overwrite_ok=True)
    func_num_statistic = workbook.add_sheet('func_num_statistic', cell_overwrite_ok=True)
    func_num_statistic.write(0, 0, 'GoID')
    func_num_statistic.write(0, 1, 'total')
    num = 1
    for key,value in func_static.items():
        func_num_statistic.write(num, 0, key)
        func_num_statistic.write(num, 1, value)
        num = num + 1

    workbook.save(f_path)



if __name__ == '__main__':
    fu_sta = func_static()
    w_to_exc(fu_sta)
    pass