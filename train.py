from file_actions import *

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

if __name__ == '__main__':
    eng_trained_dataset = load_data('train.en')
    vie_trained_dataset = load_data('train.vi')

    t_eng_vie = init_uniform_prob(eng_trained_dataset, vie_trained_dataset)
    update_trans_prob(t_eng_vie, eng_trained_dataset, vie_trained_dataset, 10000)
    write_data('t_eng_vi.txt', t_eng_vie)
