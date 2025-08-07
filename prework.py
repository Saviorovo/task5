import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

def get(data):
    poems=[]
    cur=''
    for it in data:
        if it=='\n':
            if cur:
                poems.append(str(cur))
                cur=''
        else:
            if it[-1]=='\n':
                cur+=it[:-1]
            else:
                cur+=it
    if cur:
        poems.append(str(cur))
    return poems

class Poem():
    def __init__(self,data):
        self.data=get(data)
        self.data.sort(key=lambda x: len(x))
        self.matrix=list()
        self.len_words=3
        self.word_dict={'<pad>':0,'<begin>':1,'<end>':2}
        self.tag_dict={0:'<pad>',1:'<begin>',2:'<end>'}
        #self.trained_dict = dict()
        # if trained_dict:
        #     self.trained_dict = trained_dict
        # self.embedding=[]
    def get_id(self):
        # self.embedding.append([0]*50)
        for it in self.data:
            for x in it:
                if x not in self.word_dict:
                    self.word_dict[x]=len(self.word_dict)
                    self.tag_dict[len(self.word_dict)-1]=x
        self.len_words=len(self.word_dict)

        for it in self.data:
            self.matrix.append([self.word_dict[x] for x in it])

class ClsDataset(Dataset):
    def __init__(self,data):
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        return self.data[index]

def collate_fn(batch_data):
    data=batch_data
    data=[torch.LongTensor([1,*d]) for d in data]
    pad_data=pad_sequence(data,batch_first=True,padding_value=0)
    pad_data=[torch.cat([d,torch.LongTensor([2])]) for d in pad_data]
    pad_data=list(map(list,pad_data))
    return torch.LongTensor(pad_data)

def get_batch(data,batch_size):
    dataset=ClsDataset(data)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False,drop_last=True,collate_fn=collate_fn)
    return dataloader





