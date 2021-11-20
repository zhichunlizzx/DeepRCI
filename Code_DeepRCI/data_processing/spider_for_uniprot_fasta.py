import requests

def read_uniprot_code():
    uniprot_code = []
    # file_path = r'E:\data\uniprot_code.txt'
    file_path = r'./uniprot_code.txt'
    with open(file_path, mode='r') as file_obj:
        fasta = file_obj.readlines()
        file_obj.close()
    for data in fasta:
        print(data[10:]+'are reading')
        if data == fasta[-1]:
            uniprot_code.append((data[10:]))
            break
        uniprot_code.append(data[10:-1])
    return uniprot_code


def spider():
    header = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36"}
    # cookies = {}
    file_path_write = "./0005524.txt"
    uniprot_code = read_uniprot_code()
    for uni_code in uniprot_code:
        print(uni_code)
        url = 'http://www.uniprot.org/uniprot/%s.fasta'% uni_code
        print(url)
        response = requests.session()
        response.headers.update(header)
        response1 = response.get(url)
        print(response1.text)
        with open(file_path_write, mode='a') as file_obj:
            file_obj.write(response1.text)




if __name__ == '__main__':
    spider()
    # read_uniprot_code()