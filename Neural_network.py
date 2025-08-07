import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(
            self,
            len_feature,
            len_hidden,
            len_words,
            word_dict,
            tag_dict,
            strategy='LSTM',
            pad_id=0,
            begin_id=1,
            end_id=2,
            drop_out=0.3
    ):
        super(Net, self).__init__()
        self.len_feature = len_feature
        self.len_hidden = len_hidden
        self.len_words = len_words
        self.word_to_num = word_dict
        self.num_to_word = {i: w for w, i in word_dict.items()}
        self.pad_id = pad_id
        self.begin_id = begin_id
        self.end_id = end_id
        self.strategy = strategy

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        weight = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
        self.embedding = nn.Embedding(len_words, len_feature, _weight=weight)

        if strategy == 'LSTM':
            self.cal = nn.LSTM(
                input_size=len_feature,
                hidden_size=len_hidden,
                batch_first=True,
                bidirectional=False
            )
        elif strategy == 'BILSTM':
            self.cal = nn.LSTM(
                input_size=len_feature,
                hidden_size=len_hidden,
                batch_first=True,
                bidirectional=True
            )
        elif strategy == 'GRU':
            self.cal = nn.GRU(
                input_size=len_feature,
                hidden_size=len_hidden,
                batch_first=True,
                bidirectional=False
            )
        elif strategy == 'BIGRU':
            self.cal = nn.GRU(
                input_size=len_feature,
                hidden_size=len_hidden,
                batch_first=True,
                bidirectional=True
            )
        else:
            raise ValueError('Strategy error: ' + strategy)

        out_dim = len_hidden * (2 if 'BI' in strategy else 1)
        self.fc = nn.Linear(out_dim, len_words)

        self.drop_out = nn.Dropout(drop_out)

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        emb = self.embedding(x)
        emb = self.drop_out(emb)
        self.cal.flatten_parameters()
        output, _ = self.cal(emb)
        logits = self.fc(output)
        return logits

    def Generate(self, max_len, num_sentence, random=True, head=None):
        init = torch.randn if random else torch.zeros
        num_layers = 1
        num_dirs = 2 if 'BI' in self.strategy else 1
        shape = (num_layers * num_dirs, 1, self.len_hidden)

        def init_states():
            hn = init(shape).to(self.device)
            if 'LSTM' in self.strategy:
                cn = init(shape).to(self.device)
                return hn, cn
            return hn, None

        self.cal.flatten_parameters()
        start_token = torch.tensor([self.begin_id], dtype=torch.long, device=self.device)
        end_id = self.end_id
        period_id = self.word_to_num.get('。', end_id)
        poem = []

        if head is None:
            hn, cn = init_states()
            x = start_token
            while len(poem) < num_sentence:
                seq = []
                for _ in range(max_len):
                    emb = self.embedding(x).view(1, 1, -1)
                    if 'LSTM' in self.strategy:
                        out, (hn, cn) = self.cal(emb, (hn, cn))
                    else:
                        out, hn = self.cal(emb, hn)
                    logits = self.fc(out)
                    token = logits.topk(1, dim=-1)[1].item()
                    x = torch.tensor([token], dtype=torch.long, device=self.device)
                    if token == end_id:
                        x = start_token
                        break
                    seq.append(self.num_to_word[token])
                    if token == period_id:
                        break
                else:
                    x = torch.tensor([period_id], dtype=torch.long, device=self.device)
                if seq:
                    poem.append(seq)
            return poem

        for w in head:
            if w not in self.word_to_num:
                raise ValueError(f"Word '{w}' not in dictionary.")

        for w in head:
            token_id = self.word_to_num[w]
            sentence = [w]
            hn, cn = init_states()
            x = torch.tensor([token_id], dtype=torch.long, device=self.device)
            for _ in range(max_len - 1):
                emb = self.embedding(x).view(1, 1, -1)
                if 'LSTM' in self.strategy:
                    out, (hn, cn) = self.cal(emb, (hn, cn))
                else:
                    out, hn = self.cal(emb, hn)
                logits = self.fc(out)
                token = logits.topk(1, dim=-1)[1].item()
                x = torch.tensor([token], dtype=torch.long, device=self.device)
                sentence.append(self.num_to_word.get(token, '<unk>'))
                if token == period_id:
                    break
            poem.append(sentence)
        return poem
