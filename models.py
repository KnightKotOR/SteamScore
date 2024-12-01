import torch
import torch.nn as nn
import time


class UniLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_units, output_size, dropout, seq_len, batch_size, num_layers=2):
        super(UniLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.seq_len = seq_len
        self.num_units = num_units
        self.batch_size = batch_size
        self.rnn = torch.nn.LSTM(embed_size, num_units, num_layers, bidirectional=False)
        self.dense = torch.nn.Linear(seq_len * num_units, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, target):
        embed_input = self.embedding(x)
        embed_input = self.dropout(embed_input)
        output, hidden = self.rnn(embed_input)
        logits = self.dense(output.reshape([-1, self.seq_len * self.num_units]))
        prediction = torch.sigmoid(logits)
        return prediction


class QRNNLayer(nn.Module):
    def __init__(self, batch_size, input_size, n_filters, kernel_size, embed_size, device, dropout):
        super(QRNNLayer, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.embed_size = embed_size
        self.dropout = torch.nn.Dropout(dropout)
        self.device = device
        self.conv1 = torch.nn.Conv1d(self.input_size, self.n_filters, self.kernel_size)
        self.conv2 = torch.nn.Conv1d(self.input_size, self.n_filters, self.kernel_size)
        self.conv3 = torch.nn.Conv1d(self.input_size, self.n_filters, self.kernel_size)

    def get_description(self):
        print('batch size: ', self.batch_size)
        print('input size: ', self.input_size)
        print('n_filters: ', self.n_filters)
        print('kernel size: ', self.kernel_size)
        print('embed size: ', self.embed_size)
        print('device: ', self.device)

    def forward(self, masked_input, h, c):
        z, f, o = self.masked_conv(masked_input)
        h, c = self.pool(c, z, f, o)
        masked_input = h
        return masked_input, h, c

    def masked_conv(self, x):
        if len(x.shape) == 2:
            pad = torch.zeros([1, self.input_size], device=self.device)
            x = torch.cat([pad, x], 0).permute(1, 0)
            z = torch.tanh((self.conv1(x)))
            f = torch.sigmoid((self.conv2(x)))
            o = torch.sigmoid((self.conv3(x)))
            one_mask = torch.ones_like(f, device=self.device) - f
            f = 1 - self.dropout(one_mask)
            return z.permute(1, 0), f.permute(1, 0), o.permute(1, 0)
        pad = torch.zeros([self.batch_size, 1, self.input_size], device=self.device)
        x = torch.cat([pad, x], 1).permute(0, 2, 1)
        z = torch.tanh((self.conv1(x)))
        f = torch.sigmoid((self.conv2(x)))
        o = torch.sigmoid((self.conv3(x)))
        one_mask = torch.ones_like(f, device=self.device) - f
        f = 1 - self.dropout(one_mask)
        return z.permute(0, 2, 1), f.permute(0, 2, 1), o.permute(0, 2, 1)

    def pool(self, prev_c, z, f, o):
        c = torch.mul(f, prev_c) + torch.mul(1 - f, z)
        h = torch.mul(o, c)
        return h, c


class QRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, n_filters, kernel_size, batch_size, seq_len, layers, device, dropout):
        super(QRNN, self).__init__()
        self.embed_size = embed_size
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_layer = layers
        self.device = device
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.dense = torch.nn.Linear(self.seq_len * self.n_filters, 1)
        self.QRNN_layers = torch.nn.ModuleList([QRNNLayer(self.batch_size, embed_size if layer == 0 else n_filters,
                                                          self.n_filters, self.kernel_size, self.embed_size,
                                                          self.device,
                                                          dropout, ) for layer in range(self.num_layer)])

    def get_description(self):
        for layer in self.QRNN_layers:
            layer.get_description()

    def forward(self, x, target):
        x = self.embedding(x)
        h = torch.zeros([self.batch_size, self.seq_len, self.n_filters], device=self.device)
        c = torch.zeros_like(h, device=self.device)

        masked_input = x
        for l, layer in enumerate(self.QRNN_layers):
            masked_input, h, c = layer(masked_input, h, c)
        dense_input = h.reshape([self.batch_size, -1])
        logits = self.dense(dense_input)
        prediction = torch.sigmoid(logits)
        return prediction

    def predict(self, x):
        x = self.embedding(x)
        h = torch.zeros([self.batch_size, self.seq_len, self.n_filters], device=self.device)
        c = torch.zeros_like(h, device=self.device)

        masked_input = x
        for l, layer in enumerate(self.QRNN_layers):
            masked_input, h, c = layer(masked_input, h, c)
        dense_input = h.reshape([self.batch_size, -1])
        logits = self.dense(dense_input)
        prediction = torch.sigmoid(logits)
        return prediction


class UniGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, num_units, output_size, dropout, seq_len, batch_size, layers):
        super(UniGRU, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_units = num_units
        self.rnn = torch.nn.GRU(embed_size, num_units, num_layers=layers, bidirectional=False)
        self.dense = torch.nn.Linear(seq_len * num_units, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, target):
        embed_input = self.embedding(x)
        embed_input = self.dropout(embed_input)
        output, hidden = self.rnn(embed_input)
        logits = self.dense(output.view([-1, self.seq_len * self.num_units]))
        prediction = torch.sigmoid(logits)
        return prediction


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_units, output_size, dropout, seq_len, batch_size, num_layers):
        super(BiLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_units = num_units
        self.rnn = torch.nn.LSTM(embed_size, num_units, num_layers, bidirectional=True)
        self.dense = torch.nn.Linear(seq_len * num_units * 2, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, target):
        embed_input = self.embedding(x)
        embed_input = self.dropout(embed_input)
        output, hidden = self.rnn(embed_input)

        logits = self.dense(output.view([-1, self.seq_len * self.num_units * 2]))
        prediction = torch.sigmoid(logits)
        return prediction