def find_max_eng_vie_prob(t_prob, vie_word):
    '''
        -   Hàm tìm từ tiếng anh có xác suất cao nhất khi được dịch từ một từ tiếng việt
        -   Trả về cặp từ có xác suất cao nhất và giá trị xác suất của cặp từ đó
        -   Trong trường hợp không tìm thấy từ tiếng việt (chưa được training), thêm cặp
            từ tiếng việt đó vào trained_data và trả về cặp từ đó với xác suất bằng 1
    '''
    max_prob = -1
    max_prob_word_pair = tuple()
    for line in t_prob:
        if vie_word in line:
            if float(line[2]) > max_prob:
                max_prob = float(line[2])
                max_prob_word_pair = (line[0], line[1])
    if max_prob == -1:
        # t_prob[(vie_word, vie_word)] = 1
        max_prob_word_pair = (vie_word, vie_word)
        max_prob = 1.0
    return max_prob_word_pair, max_prob

def translate_eng_vie_sentence(t_prob, vie_sen):
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
        prob_eng_vie[max_prob_word_pair] = max_prob
    return prob_eng_vie