import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import to_var


class DynamicVAE(nn.Module):

    def __init__(self, rnn_type, hidden_size, latent_size, encoder_embedding_size, output_embedding_size,
                 decoder_embedding_size, num_layers=1, bidirectional=False, variable_length=False, **params):
        super().__init__()
        self.latent_size = latent_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.variable_length = variable_length
        rnn = eval('nn.' + rnn_type.upper())

        self.encoder_rnn = rnn(encoder_embedding_size, hidden_size, num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(decoder_embedding_size, hidden_size, num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2log_v = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2embedding = nn.Linear(hidden_size * (2 if bidirectional else 1), output_embedding_size)
        self.mean2latent = nn.Sequential(nn.Linear(latent_size, int(hidden_size / 2)), nn.ReLU(),
                                         nn.Linear(int(hidden_size / 2), 1))

    def forward(self, input_sequence, encoder_filter, decoder_filter, seq_lengths, noise_scale=1.0):
        batch_size = input_sequence.size(0)
        en_input_sequence = encoder_filter(input_sequence)
        en_input_embedding = en_input_sequence.to(torch.float32)
        if self.variable_length:
            en_input_embedding = pack_padded_sequence(en_input_embedding, seq_lengths, batch_first=True)
        output, hidden = self.encoder_rnn(en_input_embedding)
        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        mean = self.hidden2mean(hidden)
        log_v = self.hidden2log_v(hidden)
        std = torch.exp(0.5 * log_v)
        mean_pred = self.mean2latent(mean)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        if self.training:
            z = z * std * noise_scale + mean
        else:
            z = mean
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        de_input_sequence = decoder_filter(input_sequence)
        de_input_embedding = de_input_sequence.to(torch.float32)
        if self.variable_length:
            de_input_embedding = pack_padded_sequence(de_input_embedding, seq_lengths, batch_first=True)

            outputs, _ = self.decoder_rnn(de_input_embedding, hidden)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, _ = self.decoder_rnn(de_input_embedding, hidden)
        log_p = self.outputs2embedding(outputs)
        return log_p, mean, log_v, z, mean_pred


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class DynamicVAE_plus(nn.Module):

    def __init__(self, rnn_type, hidden_size, latent_size, encoder_embedding_size, output_embedding_size,
                 decoder_embedding_size, num_layers=1, bidirectional=False, variable_length=False, **params):
        super().__init__()
        self.latent_size = latent_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.variable_length = variable_length
        self.output_embedding_size = output_embedding_size
        rnn = eval('nn.' + rnn_type.upper())

        self.encoder_rnn = rnn(encoder_embedding_size, hidden_size, num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2log_v = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2embedding = nn.Linear(hidden_size * (2 if bidirectional else 1), output_embedding_size)
        self.mean2latent = nn.Sequential(nn.Linear(latent_size, int(hidden_size / 2)), nn.ReLU(),
                                         nn.Linear(int(hidden_size / 2), 1))
        self.combine_linear = nn.Linear(encoder_embedding_size, 10)
        self.decoder_rnn = rnn(10, hidden_size, num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)
        self.dropout_layer = nn.Dropout(p=0.5)

        self.encoder_conv_layer = ConvLayer(encoder_embedding_size, 7)

    def forward(self, input_sequence, encoder_filter, decoder_filter, seq_lengths, noise_scale=1.0):
        hidden_to_tuple = False

        batch_size = input_sequence.size(0)
        en_input_sequence = encoder_filter(input_sequence)
        en_input_embedding = en_input_sequence.to(torch.float32)  # torch.Size([batchsize, window length, feature size])
        en_input_embedding = self.encoder_conv_layer(en_input_embedding)
        if self.variable_length:
            en_input_embedding = pack_padded_sequence(en_input_embedding, seq_lengths, batch_first=True)
        output, hidden = self.encoder_rnn(en_input_embedding)
        if type(hidden) is tuple:  # for lstm and rnn difference
            hidden_to_tuple = True
            hidden = hidden[0]
        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        mean = self.hidden2mean(hidden)
        log_v = self.hidden2log_v(hidden)
        std = torch.exp(0.5 * log_v)
        mean_pred = self.mean2latent(mean)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        if self.training:
            z = z * std * noise_scale + mean
        else:
            z = mean
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        if hidden_to_tuple:
            if self.bidirectional:
                hidden = (hidden, torch.zeros((2 * self.num_layers, batch_size, self.hidden_size)).to(hidden.device))
            else:
                hidden = (hidden, torch.zeros((1 * self.num_layers, batch_size, self.hidden_size)).to(hidden.device))
        de_input_sequence = decoder_filter(input_sequence)  # torch.Size([batchsize, 128, decoder data dim])
        de_input_embedding = de_input_sequence.to(torch.float32)

        if self.variable_length:
            de_input_embedding = pack_padded_sequence(de_input_embedding, seq_lengths, batch_first=True)

            outputs, _ = self.decoder_rnn(de_input_embedding, hidden)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        else:
            delay_signal = torch.cat((torch.zeros((batch_size, 1, self.output_embedding_size)).to(en_input_sequence.device), en_input_embedding[:, 1:, -self.output_embedding_size:]), dim=1)
            combine_embedding = torch.cat((de_input_embedding, delay_signal), dim=2)
            combine_embedding = self.combine_linear(combine_embedding)
            combine_embedding = self.dropout_layer(combine_embedding)
            outputs, _ = self.decoder_rnn(combine_embedding, hidden)
        log_p = self.outputs2embedding(outputs)
        return log_p, mean, log_v, z, mean_pred
