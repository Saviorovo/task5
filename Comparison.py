import torch
from torch import optim
import torch.nn.functional as F
import random as rd
from Neural_network import Net
from prework import get_batch
import matplotlib.pyplot as plt

# 支持的 RNN 策略列表
strategy = ['LSTM', 'BILSTM', 'GRU', 'BIGRU']

perplexity_records, models = [], []

def train_model(model, train, lr=0.001, epochs=100):

    # 固定随机数以保证可复现
    rd.seed(2025)
    torch.cuda.manual_seed(2025)
    torch.manual_seed(2025)

    device = model.device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fun = F.cross_entropy
    model.train()

    # 用于存放本模型的 perplexity 序列
    model_ppl = []

    for epoch in range(1, epochs + 1):
        total_loss = torch.tensor(0.0, device=device)
        # 遍历所有 batch
        for batch in train:
            # 准备输入和目标，移动到设备上
            x = batch
            x_input = x[:, :-1].to(device)
            y_target = x[:, 1:].to(device)

            optimizer.zero_grad()
            pred = model(x_input).transpose(1, 2)  # (batch, vocab_size, seq_len)
            loss = loss_fun(pred, y_target)
            total_loss += loss / x_input.size(1)
            loss.backward()
            optimizer.step()

        # 计算平均交叉熵损失，并换算 perplexity
        avg_loss = total_loss / len(train)        # 张量，device 上
        ppl = torch.exp(avg_loss)                # perplexity 张量，也在 device 上

        # 记录并打印
        model_ppl.append(ppl.item())             # 转为 Python 数值存储
        print(f"---------- Epoch {epoch} ----------")
        print(f"Perplexity: {ppl.item():.4f}")

    perplexity_records.append(model_ppl)
    models.append(model)


def NN_plot(poem_matrix,len_hidden, len_words, word_dict, tag_dict, batch_size,
            lr=0.001, epochs=50):
    train_data = get_batch(poem_matrix,batch_size)

    # 循环策略，训练每个模型
    for strat in strategy:
        print(f"\n=== Training strategy: {strat} ===")
        model = Net(
            len_feature=50,
            len_hidden=len_hidden,
            len_words=len_words,
            word_dict=word_dict,
            tag_dict=tag_dict,
            strategy=strat
        )
        train_model(model, train_data, lr=lr, epochs=epochs)

    M=models[2]

    def cat_poem(l):
        poem = list()
        for item in l:
            poem.append(''.join(item))
        return poem
    print('生成固定诗句')
    poem=cat_poem(M.Generate(16,4,random=False))
    for sent in poem:
        print(sent)

    print('生成随机诗句')
    torch.manual_seed(2025)
    poem = cat_poem(M.Generate(16,4,random=True))
    for sent in poem:
        print(sent)

    print('生成固定藏头诗')
    poem = cat_poem(M.Generate(max_len=20,num_sentence=4,random=False,head="春夏秋冬"))
    for sent in poem:
        print(sent)

    print('生成随机藏头诗')
    torch.manual_seed(2025)
    poem = cat_poem(M.Generate(max_len=20,num_sentence=4,random=True,head="春夏秋冬"))
    for sent in poem:
        print(sent)

    # 绘制 perplexity 曲线
    plt.figure(figsize=(10, 6))
    for ppl_list, strat in zip(perplexity_records, strategy):
        epochs_range = range(1, len(ppl_list) + 1)
        plt.plot(epochs_range, ppl_list, marker='o', label=strat)

    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs. Epoch for Different RNN Strategies')
    plt.xticks(range(1, epochs + 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.show()

