import io
#!/usr/bin/python
# -*- coding: utf8 -*-

# Function to read input file
def load_data(filename):
    with io.open(filename,'r',encoding='utf8') as f:
        content = f.read().splitlines()
    f.close()
    dataset = list(content)
    for i in range(0,len(dataset)):
        lines_split = dataset[i].lower().split()
        dataset[i] = lines_split
    return dataset

# Function to write output file
def write_data(filename, dataset):
    with io.open(filename, 'w',encoding='utf8') as f:  
        for keys, value in dataset.items():  
            f.write('%s\t%s\t%s\n' % (keys[0], keys[1], value))
    f.close()