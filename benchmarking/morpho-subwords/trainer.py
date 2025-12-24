import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import os, copy, json
from tqdm import tqdm

from data_loader import subword_loader, Vocab
from model import WordLSTM


# e-commerce max subwrds 800, movies 2500

tasks = {
"movies": (10, "movies"),
"e-commerce": (5, "e-commerce")
}

class Trainer:
  def __init__(self, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs('ckp', exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_name = args.dataset
    num_classes, config_name = tasks[dataset_name]

    self.vocab_obj = Vocab(args.vocab, args.subwords_cache)
    self.vocab_size = self.vocab_obj.vocab_size()

    train_set =  load_dataset("turkish-nlp-suite/SentiTurca", config_name, split="train")
    test_set =  load_dataset("turkish-nlp-suite/SentiTurca", config_name, split="test")
    valid_set =  load_dataset("turkish-nlp-suite/SentiTurca", config_name, split="validation")

    train_loader = subword_loader(train_set, self.vocab_obj, max_len=args.max_len, batch_size=args.batch_size)
    test_loader = subword_loader(test_set, self.vocab_obj, max_len=args.max_len, batch_size=args.val_batch_size)
    valid_loader = subword_loader(valid_set, self.vocab_obj, max_len=args.max_len, batch_size=args.val_batch_size)

    self.visualize_file = "movs_visual.file"
    self.make_visualize = args.visualize

    

    print('Done loading data\n')

    if torch.cuda.device_count() > 0:
      print(f"{torch.cuda.device_count()} GPUs found")

    print('Initializing model....')
    model = WordLSTM(args.hidden_dim, args.embed_dim, self.vocab_size, num_classes, with_attn=args.with_attn)

    model = nn.DataParallel(model)
    model.to(device)
    params = model.parameters()

    optimizer = AdamW(params, lr=args.lr, weight_decay=5e-4)
    criterion = nn.NLLLoss()
    self.device = device
    self.model = model
    self.optimizer = optimizer
    self.criterion = criterion
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.valid_loader = test_loader
    self.args = args
    self.epoch_accuracies = []
    self.all_losses = []

  def train(self):
    best_epoch = 0
    print("First epoch will start soon")
    for epoch in range(self.args.epochs):
      print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
      loss = self.train_epoch()
      epoch_acc = self.eval()
      print(f'Epoch Acc: {epoch_acc:.03f}')
      self.epoch_accuracies.append(round(epoch_acc, 3)) # append epoch accuracies
      self.all_losses.append(loss) # append epoch losses
    print("Training finished here are epoch accuracies")
    with open("epoch_acc.txt", "w") as ofile:
      for pacc in self.epoch_accuracies:
        ofile.write(str(pacc) + "\n")
    print("Now all losses")
    with open("losses.txt", "w") as ofile:
      for ploss in self.all_losses:
        ofile.write(str(ploss) + "\n")
    self.final_eval()
    if self.make_visualize:
      self.visualize()

  def train_epoch(self):
    self.model.train()
    epoch_loss = 0
    for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
        self.optimizer.zero_grad()
        ids = batch['ids'].to(self.device) #input
        labels = batch["labels"].to(self.device)  # target
        label_preds, _ = self.model(ids)
        label_preds = label_preds.squeeze(dim=0)

        loss = self.criterion(label_preds, labels)
        
        loss.backward()
        self.optimizer.step()
        interval = max(len(self.train_loader) // 20, 1)
        if i % interval == 0 or i == len(self.train_loader) - 1:
            loss = round(loss.item(), 3)
            print(f'Batch: {i + 1}/{len(self.train_loader)}\ttotal loss: {loss:.3f}')
        epoch_loss += loss
    return epoch_loss / len(self.train_loader)

  def eval(self):
    self.model.eval()
    label_pred = []
    label_true = []

    loader = self.valid_loader 

    with torch.no_grad():
      for i, batch in enumerate(loader):
        ids = batch['ids'].to(self.device)
        labels = batch['labels']
        label_preds, _ = self.model(ids)
        #print(label_preds)
        label_pred += label_preds.detach().to('cpu').squeeze(dim=0).argmax(dim=-1).tolist()

        #labels = labels.reshape(-1)
        label_true += labels.detach().to('cpu').tolist()
                
    label_acc = accuracy_score(label_true, label_pred)
    return label_acc

  def final_eval(self):
    self.model.eval()
    label_pred = []
    label_true = []

    loader = self.test_loader 

    with torch.no_grad():
      for i, batch in enumerate(loader):
        ids = batch['ids'].to(self.device)
        labels = batch['labels']
        label_preds, _ = self.model(ids)
        #print(label_preds)
        label_pred += label_preds.detach().to('cpu').squeeze(dim=0).argmax(dim=-1).tolist()

        #labels = labels.reshape(-1)
        label_true += labels.detach().to('cpu').tolist()

    label_acc = accuracy_score(label_true, label_pred)
    report = precision_recall_fscore_support(label_true, label_pred, average='macro')
    print(label_acc, "Final label accuracy")
    print(report, "Classification report final")

  def visualize(self):
    sentences = open(self.visual_file).read().split("\n")
    sentences = list(filter(None, sentences))
    self.model.eval()
    with torch.no_grad():
      with open("weights.txt", "w") as ofile:
        for sentence in sentences:
          ids = self.vocab_obj.text_to_ids(sentence)
          length = len(ids)
          ids = torch.tensor(ids).unsqueeze(dim=0)
          print(ids.shape)
          label, weights = self.model(ids)
          label = label.detach().to("cpu")
          label = label.argmax(dim=-1).item()
          weights = weights.detach().to("cpu").tolist()
          ofile.write(str(length) + " " + str(label) + "\n")
          weights = weights[0]
          for dim in weights:
              dim = round(dim, 3)
              ofile.write(str(dim) + " ")
          ofile.write("\n==================\n")







if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=48, help='batch size of training')
    parser.add_argument('--val_batch_size', type=int, default=4, help='batch size of validating and testing')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--gpu', type=str, default='', help='GPUs to use')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_len', type=int, default=1024, help='Maximum number of chars per example')
    parser.add_argument('--with_attn', type=bool, default=False, help='whether you want attention on top of the LSTM')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dim of LSTM')
    parser.add_argument('--embed_dim', type=int, default=50, help='Embedding dimension per character')
    parser.add_argument('--dataset', type=str, help='Either movies of e-commerce')
    parser.add_argument('--visualize', type=bool, default=False, help='Dump attention weights')
    parser.add_argument('--vocab', type=str, help='Pllace of the vocab file to be used')
    parser.add_argument('--subwords_cache', type=str, help='Pllace of the vocab file to be used')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(args)
    engine = Trainer(args)
    engine.train()



