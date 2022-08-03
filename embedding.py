from unittest import result
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import requests
import random
import itertools
import math

# SQL and ORM
import sqlite3
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData
from sqlalchemy.sql import select
from sqlalchemy import text

# Get champion list w/ id
URL_CHAMPION_INFO = 'http://ddragon.leagueoflegends.com/cdn/12.14.1/data/en_US/champion.json'
champion_info_response = requests.get(URL_CHAMPION_INFO)

ID_TO_CHAMP = {}
for champName in champion_info_response.json()['data'].keys():
    ID_TO_CHAMP[int(champion_info_response.json()['data'][champName]['key'])] = champName.lower()

CHAMP_TO_INDEX = {}
CHAMP_TO_INDEX = dict(zip(ID_TO_CHAMP.values(),range(162)))

print(len(list(ID_TO_CHAMP.values())))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, summonersMasteryList):
        super().__init__()
        self.summonersPointList = summonersMasteryList
        
    def __len__(self):
        return len(self.summonersPointList)
    
    def __getitem__(self, index):
        result = []
        pointDict = self.summonersPointList[index]
        champComb = self._champion_choice(pointDict)
        for comb in champComb:
            champ1Point = pointDict[comb[0]]
            champ2Point = pointDict[comb[1]]
            if champ1Point is None or champ2Point is None:
                continue
            similarity = math.sqrt(champ1Point * champ2Point) / pointDict['sum'] # normalize point?
            # (champ1 index, champ2 index, dot product) Not champion id!!
            result.append((CHAMP_TO_INDEX[comb[0]],CHAMP_TO_INDEX[comb[1]],similarity))
            
        result = torch.tensor(result)
        return result
    
    # Select champion pair with weight
    def _champion_choice(self,champPointDict):
        # Give weight according to the mastery point
        pointList = [0 if x==None else pow(int(x),0.7) for x in list(champPointDict.values())[2:]]
        # Select 15 champions
        champWeightedList = random.choices(list(ID_TO_CHAMP.values()), weights=pointList, k=15)
        champComb = list(itertools.combinations(champWeightedList, 2)) # list of tuple (champName, champName) <- string
        return champComb
    
class Model(nn.Module):
    def __init__(self, nChampion, nEmbedding):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(nChampion, nEmbedding)
        
    def forward(self, champ1, champ2): # champ1 : (batchsize, 1)
        # champ1Embedding = torch.unsqueeze(self.embedding(champ1),1)
        # champ2Embedding = torch.unsqueeze(self.embedding(champ2),2)
        dotResult = torch.bmm(torch.unsqueeze(self.embedding(champ1),1),
                              torch.unsqueeze(self.embedding(champ2),2))
        return dotResult
    
# Get summoner's mastery points    
masteryEngine = create_engine('sqlite:///summoner_mastery.db', echo=False)
connMastery = masteryEngine.connect()
meta = MetaData()
meta.reflect(bind = masteryEngine)
sel = select(meta.tables['summoner_mastery_table']) # get all column if blank 
masteryResult = connMastery.execute(sel)
summonerMasteryResult = []
for elem in masteryResult:
    summonerMasteryResult.append(dict(elem))
    
######

# Training step
train_dataset = Dataset(summonerMasteryResult)
training_generator = torch.utils.data.DataLoader(train_dataset, shuffle=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(f'Torch device: {device}')

net = Model(161,15)#.cuda()
net.train()
optimizer = optim.Adam(net.parameters())
total_epoch = 2000
iteration = 0
total_iter = total_epoch*train_dataset.__len__()
# what I learn:

for epoch in range(total_epoch):
    for account in training_generator:
        print(account.shape)
        optimizer.zero_grad()
        champ1 = account[0,:,0].to(torch.int) # champ1 index list
        champ2 = account[0,:,1].to(torch.int) # champ2 index list
        gt_score = account[0,:,2] # champion dot product
        pred_score = net(champ1,champ2).view(-1,)
        loss = F.mse_loss(pred_score, gt_score)
        iteration += 1
        print(f'Iter: {iteration:06d} / {total_iter:06d} | Loss: {loss.item():.4f}')
        loss.backward()
        optimizer.step()

# Save model
torch.save({'model_state_dict': net.state_dict()}, 'embedding.pt')
    
# Embedding vector visualization
for param in net.parameters():
    print(type(param), param.size())
    champ_embedding = param.cpu().detach().numpy()
    break
    
champ_embedding = champ_embedding / np.linalg.norm(champ_embedding, axis = 1).reshape(-1, 1)
print(champ_embedding.shape)
ahri = (np.expand_dims(champ_embedding[1],0) @ champ_embedding.T)[0]

ahri_dict = dict(zip(ID_TO_CHAMP.values(),ahri))
ahri_dict = dict(sorted(ahri_dict.items(), key=lambda item: item[1], reverse=True))
print(ahri_dict)
