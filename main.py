import random as rd
import os,csv
from prework import Poem,get_batch
from Comparison import NN_plot

strategy=['LSTM','BILSTM','GRU','BIGRU']
file_path=os.path.join('poetryFromTang.txt')
with open(file_path,'r',encoding='utf-8') as f:
    train=f.readlines()

poem=Poem(train)
poem.get_id()
#print(poem.data)
batch_size=1
lr=0.001
epochs=60

NN_plot(poem.matrix,50,poem.len_words,poem.word_dict,poem.tag_dict,batch_size,lr,epochs)




