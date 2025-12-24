import torch
import torch.nn as nn
import torch.nn.functional as F



class WordLSTM(nn.Module):
  def __init__(self, hidden_dim, embedding_dim, vocab_size, num_targets, with_attn=False):
    super(WordLSTM, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.LSTM = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=1)
    self.dense = nn.Linear(hidden_dim, num_targets)
    self.softmax = nn.LogSoftmax(dim=1)
    self.with_attn = with_attn
    self.hidden_dim = hidden_dim

  def attention_layer(self, lstm_out, final_hidden):
    hidden = final_hidden.squeeze(0)
    attention_weights = torch.bmm(lstm_out, hidden.unsqueeze(2)).squeeze(2)
    alphas = F.softmax(attention_weights, 1)
    new_hidden = torch.bmm(lstm_out.transpose(1,2), alphas.unsqueeze(2)).squeeze(2)
    new_hidden = new_hidden.unsqueeze(dim=0)
    return alphas, new_hidden

  def forward(self, ids):
    embeds = self.embedding(ids)
    lstm_out, (lstm_hidden, lstm_cell) = self.LSTM(embeds)

    bidirection_sum_1 = lstm_out[:, :, :self.hidden_dim]+ lstm_out[:, :, self.hidden_dim:]
    bidirection_sum = (lstm_hidden[-2,:,:]+lstm_hidden[-1,:,:]).unsqueeze(0)
    if self.with_attn:
      alphas, attn_hidden = self.attention_layer(bidirection_sum_1, bidirection_sum)
    else:
      alphas = None

    if self.with_attn:
       label = self.dense(attn_hidden)
    else:
      label = self.dense(bidirection_sum)

    label = self.softmax(label)
    label = label.squeeze(dim=1)
    return label, alphas


'''
ids =  torch.tensor([[1, 4, 5, 6], [1, 2, 3, 7], [0, 2, 3,4]])
char_lstm = CharLSTM(20, 10, 8, 12, with_attn=True)
label = char_lstm(ids)
print(label.shape)
'''

