import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()

        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True)

        self.liner = nn.Linear(2 * num_layers * hidden_size, num_classes)
        self.act_func = nn.Softmax(dim=1)

    def forward(self, x):
        # lstm的输入维度为 [seq_len, batch_size, input_size]
        # x [batch_size, sentence_length, embedding_size]
        x = x.permute(1, 0, 2)  # [sentence_length, batch_size, embedding_size]

        # 由于数据集不一定是预先设置的batch_size的整数倍，所以用size(1)获取当前数据实际的batch
        batch_size = x.size(1)

        # 设置lstm最初的前项输出
        h_0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to("cuda")
        c_0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to("cuda")

        # out[seq_len, batch_size, 2 * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        # h_n, c_n [num_layers * 2, batch_size, hidden_size]
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        x = h_n  # [num_layers*2, batch_size, hidden_size]
        x = x.permute(1, 0, 2)  # [batch_size, num_layers*2, hidden_size]
        x = x.contiguous().view(batch_size,
                                self.num_layers * 2 * self.hidden_size)  # [batch_size, 2*num_layers*hidden_size]
        x = self.liner(x)
        x = self.act_func(x)
        return x
