import numpy as np 
import pandas as pd 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

from sklearn.metrics import roc_auc_score

import warnings
warnings.simplefilter('ignore')

seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i, w in enumerate(seq_rdic)}

def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0]
    else:
        return [seq_dic[aa] for aa in seq]

def padding_seq(x, max_len=2500):
    if len(x) == 0:
        return x
    elif len(x) >= max_len:
        return x[:max_len]
    else:
        return x + [0]*(max_len-len(x))

def parse_data(dti_dir, prot_len=2500, drug_len=2048, is_train=True):

    protein_col = "protein"
    drug_col = "compound"
    label_col = "label"
    weight_col = "weight"
    col_names = [protein_col, drug_col, label_col, weight_col]
    
    dti_df = pd.read_csv(dti_dir, header=0)

    dti_df[protein_col] = dti_df[protein_col].map(lambda a: encodeSeq(a, seq_dic))
    
    drug_feature = np.stack(dti_df[drug_col].map(
        lambda sm: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(sm),2,nBits=drug_len)
    ))
    
    protein_feature = np.stack(dti_df[protein_col].map(padding_seq))
    label = dti_df[label_col].values
    if is_train:
        weight = dti_df[weight_col].values
    
    print("\tPositive data : %d" %(sum(dti_df[label_col])))
    print("\tNegative data : %d" %(dti_df.shape[0] - sum(dti_df[label_col])))
    
    if is_train:
        return {"protein_feature": protein_feature, 
            "drug_feature": drug_feature, 
            "label": label,
            "weight": weight,
            }
    else:
        return {"protein_feature": protein_feature, 
            "drug_feature": drug_feature, 
            "label": label,
            }

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_drug = nn.Linear(2048, 64)
        
        self.conv1D = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=128, stride=1, padding=0)
        self.batch = nn.BatchNorm1d(64)
        
        self.fc_out = nn.Linear(128, 128)
        self.fc_interaction = nn.Linear(128, 2)

    def forward(self, drug, protein):
        compound_vector = self.fc_drug(drug)
        compound_vector = torch.relu(compound_vector)

        protein = torch.unsqueeze(protein, 1)
        protein_vector = self.conv1D(protein)
        protein_vector = self.batch(protein_vector)
        protein_vector = torch.relu(protein_vector)
        
        protein_vector = F.max_pool1d(protein_vector, kernel_size=2373)
        protein_vector = torch.squeeze(protein_vector, 2)
        
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        
        for j in range(2):
            cat_vector = torch.relu(self.fc_out(cat_vector))
        out = self.fc_interaction(cat_vector)
        return out

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

def train(tl, vl, is_wight):
    model= Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scores = []
    for epoch in range(1000):
        loss_total = 0
        for data in tl:
            d, p, w, true_label = data
            d, p, w, true_label  = d.to(device), p.to(device), w.to(device), true_label.to(device)

            optimizer.zero_grad()
            output = model(d, p)
            predicted_label = nn.Softmax(dim=1)(output)
            if is_wight:
                loss = torch.mean(F.cross_entropy( predicted_label, true_label, reduce='none'))
            else:
                loss = F.cross_entropy( predicted_label, true_label, reduce='mean')
            loss.backward()
            optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        pred_y, true_y = [], []
        for data in vl:
            d, p, w, true_label = data
            d, p, w, true_label  = d.to(device), p.to(device), w.to(device), true_label.to(device)
            output = model(d, p)
            predicted_label = nn.Softmax(dim=1)(output)[:,1].to('cpu').data.numpy().tolist()
            pred_y += predicted_label
            true_y += true_label.to('cpu').data.numpy().tolist()
        score = roc_auc_score(true_y, pred_y)
        if epoch > 10:
            if np.mean(scores[-10:]) > score:
                scores.append(score)
                break
        scores.append(score)

        if epoch % 10 == 0:
            print("epoch: {}, loss: {}, AUC: {}".format(epoch, loss_total, score))
    return model

def test(tl, model):
    pred_y, true_y = [], []
    for data in tl:
        d, p, true_label = data
        d, p, true_label  = d.to(device), p.to(device), true_label.to(device)
        output = model(d, p)
        predicted_label = nn.Softmax(dim=1)(output)[:,1].to('cpu').data.numpy().tolist()
        pred_y += predicted_label
        true_y += true_label.to('cpu').data.numpy().tolist()
    score = roc_auc_score(true_y, pred_y)
    print(score)
    return score

dataset_names = ["human", "celegans"]

for dataset_name in dataset_names:
    for random_state in range(100):
        train_filename = "../../data/maked/bias/" + str(random_state) + "/train_" + dataset_name + "_link_strong.csv"
        test_filename = "../../data/maked/bias/" + str(random_state) + "/test_" + dataset_name + "_link_strong.csv"
        
        train_datas = parse_data(train_filename)
        test_datas = parse_data(test_filename, is_train=False)
        
        train_drug_dataset = torch.FloatTensor(train_datas["drug_feature"])
        train_protein_dataset = torch.FloatTensor(train_datas["protein_feature"])
        train_weight_dataset = torch.FloatTensor(train_datas["weight"])
        train_target_dataset = torch.LongTensor(train_datas["label"])
        
        test_drug_dataset = torch.FloatTensor(test_datas["drug_feature"])
        test_protein_dataset = torch.FloatTensor(test_datas["protein_feature"])
        test_target_dataset = torch.LongTensor(test_datas["label"])
        train_dataset = torch.utils.data.TensorDataset(train_drug_dataset, train_protein_dataset, train_weight_dataset, train_target_dataset)
        test_dataset = torch.utils.data.TensorDataset(test_drug_dataset, test_protein_dataset, test_target_dataset)

        N = len(train_dataset)
        train_size = int(0.8 * N)
        val_size = N - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
        
        model_baseline = train(tl=train_loader, vl=val_loader, is_wight=False)
        model_propose = train(tl=train_loader, vl=val_loader, is_wight=True)

        score_baseline = test(tl=test_loader, model=model_baseline)
        score_propose = test(tl=test_loader, model=model_propose)

        with open("./" + dataset_name+"_baseline_link_strong.txt", "a") as f:
            f.write(str(score_baseline) +"\n")
        with open("./" + dataset_name+"_propose_link_strong.txt", "a") as f:
            f.write(str(score_propose) +"\n")
