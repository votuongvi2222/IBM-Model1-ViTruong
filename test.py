from train import *
from translation import *
from file_actions import *

def compute_prob_eng_vie_sentence(translated_sentence, vie_sen_len):
    prob = 1
    for word_pair in translated_sentence.keys():
        prob *= translated_sentence[word_pair]
    return prob/((vie_sen_len+1)**vie_sen_len)

def compare_result_test_sentences(result_sentence, test_sentence):
    count = 0
    for k in range(len(test_sentence)):
        if result_sentence[k] == test_sentence[k]:
            count += 1
    return count/len(test_sentence)

if __name__ == '__main__':
    eng_tested_lines = load_file('test.en')
    vie_tested_lines = load_file('test.vi')
    trained_prob_dataset = load_data('t_eng_vi.txt')
    
    prob_translated_dataset = dict()
    prob_right_translated_dataset = dict()
    for k in range(len(vie_tested_lines)):
        vie_sen = vie_tested_lines[k]
        eng_sen = eng_tested_lines[k]
        translated_sentence_dataset = translate_eng_vie_sentence(trained_prob_dataset, vie_sen)
        result_sentence = ''
        for word_pair in translated_sentence_dataset.keys():
            print(word_pair)
            result_sentence += word_pair[0] + ' '
            print(result_sentence)
        prob1 = compute_prob_eng_vie_sentence(translated_sentence_dataset, len(vie_sen))
        prob2 = compare_result_test_sentences(result_sentence, eng_sen)

        prob_translated_dataset[(result_sentence, vie_sen)] = prob1
        prob_right_translated_dataset[(result_sentence, eng_sen)] = prob2

    write_data('prob1.txt', prob_translated_dataset)
    write_data('prob2.txt', prob_right_translated_dataset)


