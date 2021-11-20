import requests
def url_spider():
    uniprot_code = 'H2MUM3'
    file_path_write = './'+uniprot_code + '.txt'
    # url1 = 'http://www.uniprot.org'
    url = 'http://www.uniprot.org/uniprot/%s.fasta'% uniprot_code
    # url = 'https://www.baidu.com'
    response = requests.get(url)
    print(response.text)
    with open(file_path_write, mode='w') as file_obj:
        file_obj.write(response.text)

def quick_go_anno():
    url = 'https://www.uniprot.org/uniprot/Q9X055'
    response = requests.get(url)

    print(response.text)

if __name__ == '__main__':
    # url_spider()
    quick_go_anno()
    pass