##pipeline: call BERT model, feed in the encoded representations, fine-tune the bert model. 

import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

# change it with respect to the original model
from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
import torch_optimizer as optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import RobertaModel


#from apex.optimizers import FusedLAMB
# import tensorflow_addons as tfa
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Nadam


TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
    """1. encode the sentences using BERT to obtain the pooled output representation of the sentence.
       2. classify the sentence by applying dropout to the pooled-output and project it using a linear layer.
       3. adjust the model paramters depending on whether we are pre-training or fine-tuning BERT"""
    
    
class BertSentClassifier(torch.nn.Module):
    def __init__(self, config):
        super(BertSentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = RobertaModel.from_pretrained('vinai/bertweet-base')


        # pretrain mode does not require updating bert paramters.
        # Adjust parameters.
        for param in self.bert.parameters():
            if config.option == 'pretrain': # pretrain mode
                param.requires_grad = False
            elif config.option == 'finetune': # fine-tune mode
                param.requires_grad = True

        # TODO
        # 2. classify the sentence
        # 2-1. By applying Dropout layer to the pooled-output (for regularization)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # dropout prob in config(hidden / attention : 0.1)
        # 2-2. and project it using a linear layer. (Linear classification layer)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        
        
    def forward(self, input_ids, attention_mask):
        # 1. Encode the sentences using BERT to obtain the pooled output representation of the sentence/ The sentences are encoded using forward pass. 
            
        ## TODO
        ## The final bert contextualize embedding is the hidden of [CLS] token(the first token)
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # Use the [CLS] token output as the sentence representation (index 0) / extracted for sentence classification
        
        last_hidden_state = outputs['last_hidden_state']
        pooled_output = last_hidden_state[:, 0, :]
        # Applying dropout to the pooled-output
        pooled_output = self.dropout(pooled_output)

        # Classify sentence based on the [CLS] representation.
        logits = self.classifier(pooled_output)
        
        return logits


        
        
        
# create a custom Dataset Class to be used for the dataloader
class BertDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ele = self.dataset[idx]
        return ele

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        token_type_ids = torch.LongTensor(encoding['token_type_ids'])
        labels = torch.LongTensor(labels)

        return token_ids, token_type_ids, attention_mask, labels, sents

    def collate_fn(self, all_data):
        all_data.sort(key=lambda x: -len(x[2]))  # sort by number of tokens

        batches = []
        num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            data = all_data[start_idx: start_idx + self.p.batch_size]

            token_ids, token_type_ids, attention_mask, labels, sents = self.pad_data(data)
            batches.append({
                'token_ids': token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
            })

        return batches


    # create the data which is a list of (sentence, label, token for the labels)
def create_data(filename, flag='dev'):
  tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
  #tokenizer = BertTokenizer.from_pretrained('bertweet-base')
  num_labels = {}
  data = []

  with open(filename, 'r', encoding='ISO-8859-1') as file:
    reader = csv.reader(file)
    for row in reader:
        order, label, sentence = row
        sent = sentence.lower().strip()
        tokens = tokenizer.tokenize("[CLS] " + sent + " [SEP]")
        try:
          label = int(label.strip())
        except ValueError:
          print(f"Cannot convert {label} to an integer.")
        
        if label not in num_labels:
            num_labels[label] = len(num_labels)
        data.append((sent, label, tokens))

  print(f"load {len(data)} data from {filename}")
  if flag == 'train':
      return data, len(num_labels)
  else:
      return data

# perform model evaluation in terms of the accuracy and f1 score.
@staticmethod
def model_eval(dataloader, model, device):
    model.eval() # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_type_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], batch[0]['token_type_ids'], \
                                                       batch[0]['attention_mask'], batch[0]['labels'], batch[0]['sents']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    #### Load data
    # create the data and its corresponding datasets and dataloader
    train_data, num_labels = create_data(args.train, 'train')
    dev_data = create_data(args.dev, 'valid')

    train_dataset = BertDataset(train_data, args)
    dev_dataset = BertDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    #### Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    # initialize the Senetence Classification Model

    model = AutoModel.from_pretrained('vinai/bertweet-base')
    model = BertSentClassifier(config)
    model = model.to(device)

    lr = args.lr

    ## specify the optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    #optimizer = optim.RAdam(model.parameters(), lr=lr)
    #optimizer = FusedLAMB(model.parameters(), lr=lr)
    #optimizer = tfa.optimizers.LAMB(learning_rate=lr)
    #optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr, momentum=0.9, nesterov=False, decay=0.0001)

    best_dev_acc = 0
    
    ## run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            b_ids, b_type_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], batch[0]['token_type_ids'], batch[0][
                'attention_mask'], batch[0]['labels'], batch[0]['sents']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()
            

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertSentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        dev_data = create_data(args.dev, 'valid')
        dev_dataset = BertDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = create_data(args.test, 'test')
        test_dataset = BertDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
        test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            for s, t, p in zip(dev_sents, dev_true, dev_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")

        with open(args.test_out, "w+") as f:
            print(f"test acc :: {test_acc :.3f}")
            for s, t, p in zip(test_sents, test_true, test_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="/content/drive/MyDrive/ColabNotebooks/hw3/data/mass_shooting-train.csv")
    parser.add_argument("--dev", type=str, default="/content/drive/MyDrive/ColabNotebooks/hw3/data/mass_shooting-dev.csv")
    parser.add_argument("--test", type=str, default="/content/drive/MyDrive/ColabNotebooks/hw3/data/mass_shooting-test.csv")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--dev_out", type=str, default="/content/drive/MyDrive/ColabNotebooks/hw3/data/mass_shooting-dev-output.csv")
    parser.add_argument("--test_out", type=str, default="/content/drive/MyDrive/ColabNotebooks/hw3/data/mass_shooting-test-output.csv")
    parser.add_argument("--filepath", type=str, default=None)

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    if args.filepath is None:
        args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train(args)
    test(args)