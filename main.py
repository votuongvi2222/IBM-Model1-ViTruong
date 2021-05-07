import train_model
import test_model
import sys
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
        :levels --> The levels of data's size will be trained --> list()
        :time_taken_per_level --> Store the times taken for each level --> dict()
    '''
    levels = [1000, 10000, 100000]
    time_taken_per_level = dict()
    for level in levels:
        start_time = time.time()
        t_prob = train_model.train('../datasets/train.en', '../datasets/train.vi', level)
        end_time = time.time()
        time_taken_per_level[level] = end_time - start_time
    # Plot the time excuting
    myList = time_taken_per_level.items()
    myList = sorted(myList) 
    xs, ys = zip(*myList) 

    plt.plot(xs,ys,'bo-')

    # zip joins x and y coordinates in pairs
    for x,y in zip(xs,ys):

        label = 's:'+str(x)+'\nt:'+str(y)

        plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center') 

    plt.title('RUNNING TIME')
    plt.xlabel('Size of data (the number of sentence_pairs)')
    plt.ylabel('Seconds (s)')
    plt.show()