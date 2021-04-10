import csv
import io
# from collections import defaultdict
#!/usr/bin/python
# -*- coding: utf8 -*-
def loadCsv(filename):
    lines = csv.reader(open(filename, "r", encoding='utf-8'))
    '''
        -   Tách file input thành mảng các dòng, 
            mỗi dòng được lưu theo dạng cặp câu chứa trong mảng nhỏ --> dataset
        -   Sau đó duyệt từng mảng nhỏ:
            +   Mỗi cặp câu trong mảng được cắt chuỗi lưu vào hai mảng khác nhau
            +   eng_dataset --> câu tiếng anh
            +   vie_dataset --> câu tiếng việt 
        Return: hai mảng lần lượt chứa các câu tiếng anh và tiếng việt đã đượcc cắt chuỗi
    '''
    dataset = list(lines)
    eng_dataset = list()
    vie_dataset = list()
    for i in range(1, len(dataset)):
        # print(1)
        eng_dataset.append(dataset[i][0].lower().split(' '))
        vie_dataset.append(dataset[i][1].lower().split(' '))
    return eng_dataset, vie_dataset


# Function to read input file
def load_data(filename):
    with io.open(filename,'r',encoding='utf8') as f:
        content = f.read().splitlines()
    dataset = list(content)
    for i in range(0,len(dataset)):
        lines_split = dataset[i].lower().split()
        dataset[i] = lines_split
    return dataset