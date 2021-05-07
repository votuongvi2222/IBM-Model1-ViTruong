import math
import copy
import sys
from file_actions import *

def is_converged(new, old, num_of_iterations, max_num_of_iterations=9):
    '''
        This function is used to check the stop point.
        ==== The arguments ====
        :new --> the probabilities after updating --> dict()
        :old --> the probabilities before updating --> dict()
        :num_of_iterations --> the times of updating --> int
        :max_num_of_iterations --> (optional) the limit iterations --> int (default = 0)
        =======================
        returns 
            - False, if there exists at least 1 pair is not 
            converged.
            - Else, True
    '''
    min_float = 1.0e-5 # the min value 
    if num_of_iterations > max_num_of_iterations :
        return True

    for word_pair_new, word_pair_old in zip(new, old):
        if math.fabs(new[word_pair_new] - old[word_pair_old]) > min_float:
            return False

    print('converged')
    return True

def init_uniform_prob(eng_sentences, vie_sentences , init_p = 0.25):
    '''
        function initializes synchronous initial probability values
        === Agruments ===
        :eng_sentences --> The formated dataset of english sentenced --> list(list(english word))
        :vie_sentences --> The formated dataset of vietnamese sentenced --> list(list(vietnamese word))
        :init_p --> The uniform probability --> float (defaul = 0.25)
        =================
        returns
            vie_eng_trans_prob -> probability that vie_w translates to eng_w (also known as t (e | v))
    '''
    vie_eng_trans_prob = dict()
    num_lines = len(eng_sentences)
    for k in range(0, num_lines):
        '''
            - eng_sen -> array of words separated from an English sentence
            - vie_sen -> array of words separated from a Vietnamese sentence
            - For each word in a Vietnamese sentence, assign the probability of that initialization
            It can be translated from one word in English sentence with init_p (in pairs of sentences)
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
        The function that creates a counter in the form of a dict includes 2 keys,
        Each key is a dictionary denoted as follows:
        - 'eng_vie' -> count the number of Vietnamese words V
                            translated into English word E
        - 'vie' -> Count the number of times that Vietnamese word V has been translated
                        in the train data set
        === Agruments ===
        :t_prob --> The translation probabilities --> dictionary(tuple(word_pair))
        :eng_dataset --> The formated dataset of english sentenced --> list(list(english word))
        :vie_dataset --> The formated dataset of vietnamese sentenced --> list(list(vietnamese word))
        =================
        returns counter --> dictionary()
    '''
    counter = dict()
    counter['eng_vie'] = dict()
    counter['vie'] = dict()
    num_lines = len(eng_dataset)
    for k in range(0, num_lines):
        # vie_dataset[k] = [None] + vie_dataset[k]
        for vie_word in vie_dataset[k]:
            for eng_word in eng_dataset[k]:
                counter['eng_vie'][(eng_word, vie_word)] = 0
            counter['vie'][vie_word] = 0
    print('done init counter')
    return counter
    
def update_trans_prob(t_prob, eng_dataset, vie_dataset):
    '''
        Deploying under pseudo code of EM algorithm of model 1. 
        This function is to update the probability value of each word pair, 
        to calculate the probability that a Vietnamese word can be translated 
        into a word English. The loop runs until all probabilities in t_prob 
        is converged

        === Agruments ===
        :t_prob --> The translation probabilities --> dictionary(tuple(word_pair))
        :eng_dataset --> The formated dataset of english sentenced --> list(list(english word))
        :vie_dataset --> The formated dataset of vietnamese sentenced --> list(list(vietnamese word))
        =================
    '''
    n_recurr = 0 # the number of iteratons until converged
    counter_temp = init_counter(eng_dataset, vie_dataset) # init counter
    t_prob_temp = {**t_prob} # init 
    for word_pair in t_prob_temp:
        t_prob_temp[word_pair] = 1
    while not is_converged(t_prob, t_prob_temp, n_recurr):
        n_recurr += 1
        t_prob_temp = {**t_prob} # copy t_prob to compare
        # initialize
        counter = copy.deepcopy(counter_temp)
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
            if prob == 5e-324: # almost equal to zero
                prob = 0
            t_prob[word_pair] = prob

def reduce_data(data, length):
    if length < 0 or length > len(data):
        raise ValueError('The size of data is invalid')
    else:
        return data[:length]

def train(eng_file_name, vie_file_name, data_length=None):
    '''
        This function will train the datasets, then write the result in 't_eng_vi.txt'
        ==== Arguments ====
        :eng_file_name --> The name of english sentences file --> String
        :vie_file_name --> The name of vietnamese sentences file --> String
        :data_length --> The size of data will be trained
        ===================
    '''
    eng_trained_dataset = load_data(eng_file_name)
    vie_trained_dataset = load_data(vie_file_name)
    if data_length is not None:
        eng_trained_dataset = reduce_data(eng_trained_dataset, data_length)
        vie_trained_dataset = reduce_data(vie_trained_dataset, data_length)
    t_eng_vie = init_uniform_prob(eng_trained_dataset, vie_trained_dataset)
    update_trans_prob(t_eng_vie, eng_trained_dataset, vie_trained_dataset)
    write_data('t_eng_vi.txt', t_eng_vie)
