import csv
import io
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

def init_uniform_prob(eng_sentences, vie_sentences , init_p = 0.25):
    '''
        hàm khởi tạo các giá trị xác suất đồng bộ ban đầu
        -   trained_data --> dữ liệu training đã được xử lý ở hàm load_file
        -   vie_eng_trans_prob --> xác suất vie_w dịch thành eng_w (hay còn được gọi t(e|v))
        -   eng_w_times --> số lần e_word được dịch trong dữ liệu training
    '''
    vie_eng_trans_prob = dict()
    num_lines = len(eng_sentences)
    for k in range(0, num_lines):
        '''
            -   eng_sen --> mảng các từ được tách từ một câu tiếng anh
            -   vie_sen --> mảng các từ được tách từ một câu tiếng việt
            -   với mỗi từ trong câu tiếng việt, gán xác suất khởi tạo mà
                nó có thể được dịch từ một từ trong câu tiếng anh với init_p 
                (xét theo mỗi cặp câu)
        '''
        eng_sen = eng_sentences[k]
        vie_sen = vie_sentences[k]

        eng_sen_len = len(eng_sen)
        vie_sen_len = len(vie_sen)

        for vie_word in vie_sen:
            for eng_word in eng_sen:
                vie_eng_trans_prob[(eng_word, vie_word)] = init_p
    
    return vie_eng_trans_prob

def init_counter(eng_dataset, vie_dataset):
    '''
        Hàm tạo bộ đếm dưới dạng dict bao gồm 2 keys, 
        mỗi key là một dict biểu diễn như sau;
        -   'eng_vie'   --> đếm số lần từ tiếng việt V 
                            được dịch thành từ tiếng anh E
        -   'vie'   --> Đếm số lần từ tiếng việt V được dịch 
                        trong bộ train data
    '''
    counter = dict()
    counter['eng_vie'] = dict()
    counter['vie'] = dict()
    num_lines = len(eng_dataset)
    for k in range(0, num_lines):
        for vie_word in vie_dataset[k]:
            for eng_word in eng_dataset[k]:
                counter['eng_vie'][(eng_word, vie_word)] = 0
            counter['vie'][vie_word] = 0
    return counter
    
def update_trans_prob(t_prob, eng_dataset, vie_dataset, n_recurr):
    '''
        -   Triển khai theo mã giả của thuật toán EM của model 1 
        -   Hàm này nhằm update lại giá trị xác suất của mỗi cặp từ,
            để tính khả năng mà một từ tiếng việt có thể được dịch thành một từ 
            tiếng anh.
        -   Trong trường hợp khi xác suất nhỏ hơn 0, gán pp = 0. Tương tự khi xác
            suất vượt quá 1, gán p = 1.
        -   Vòng lặp chạy theo giá trị số lần ta muốn training.
    '''
    for time in range(n_recurr):
        # initialize
        counter = init_counter(eng_dataset, vie_dataset)
        count_eng_vie = counter['eng_vie']
        total_vie = counter['vie']

        for k in range(len(eng_dataset)):
            s_total_eng = dict()
            # compute normalization
            for eng_word in eng_dataset[k]:
                s_total_eng[eng_word] = 0
                for vie_word in vie_dataset[k]:
                    s_total_eng[eng_word] += t_prob[(eng_word, vie_word)]
            # collect counts
            for eng_word in eng_dataset[k]:
                for vie_word in vie_dataset[k]:
                    count_eng_vie[(eng_word, vie_word)] += t_prob[(eng_word, vie_word)] / s_total_eng[eng_word]
                    total_vie[vie_word] += t_prob[(eng_word, vie_word)] / s_total_eng[eng_word]
        # estimate probs
        for word_pair in t_prob.keys():
            prob = count_eng_vie[word_pair] / total_vie[word_pair[1]]
            # check converged
            if prob > 1:
                t_prob[word_pair] = 1
            elif prob < 0:
                t_prob[word_pair] = 0
            else:
                t_prob[word_pair] = prob

def find_max_eng_vie_prob(t_prob, vie_word):
    '''
        -   Hàm tìm từ tiếng anh có xác suất cao nhất khi được dịch từ một từ tiếng việt
        -   Trả về cặp từ có xác suất cao nhất và giá trị xác suất của cặp từ đó
        -   Trong trường hợp không tìm thấy từ tiếng việt (chưa được training), thêm cặp
            từ tiếng việt đó vào trained_data và trả về cặp từ đó với xác suất bằng 1
    '''
    max_prob = -1
    max_prob_word_pair = tuple()
    for word_pair in t_prob.keys():
        if vie_word in word_pair:
            if t_prob[word_pair] > max_prob:
                max_prob = t_prob[word_pair]
                max_prob_word_pair = word_pair
    if max_prob == -1:
        t_prob[(vie_word, vie_word)] = 1
        max_prob_word_pair = (vie_word, vie_word)
        max_prob = 1.0
    return max_prob_word_pair, max_prob

def translate(t_prob, vie_sen):
    '''
        -   Hàm tìm câu tiếng anh có khả năng được dịch từ câu tiếng việt
            xác suất cao nhất mà không quan tâm thứ tự.
    '''
    vie_words = vie_sen.lower().split(' ')
    prob_eng_vie = dict()
    for vie_word in vie_words:
        prob_word_pair = find_max_eng_vie_prob(t_prob, vie_word)
        max_prob_word_pair = prob_word_pair[0]
        max_prob = prob_word_pair[1]
        print(max_prob_word_pair)
        prob_eng_vie[max_prob_word_pair] = max_prob
    return prob_eng_vie

def main():
    eng_dataset = load_data('train.en')
    vie_dataset = load_data('train.vi')

    t_eng_vie = init_uniform_prob(eng_dataset, vie_dataset)
    update_trans_prob(t_eng_vie, eng_dataset, vie_dataset, 10000)
    
    print(translate(t_eng_vie, 'Tôi và Vi'))
    with io.open("output.txt,'w',encoding='utf8') as f:
                 for key,value in t_eng_vie.items():
                    f.write('%s\t%s\t%s\n' % (key[0],key[1],value))
    #token = io.open('output.txt','r',encoding = 'utf8')
    #linetoken = token.readlines()
    #resulttoken = []
    #for x in linetoken:
    #   resulttoken.append(x.split(" ")[0])
    #token.close()
    
main()
