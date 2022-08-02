from unittest import result
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
print(CHAMP_TO_INDEX)

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
        champCombList = random.sample(list(ID_TO_CHAMP.values()),15) #.keys()
        champComb = list(itertools.combinations(champCombList, 2))
        for comb in champComb:
            champ1Point = pointDict[comb[0]]
            champ2Point = pointDict[comb[1]]
            if champ1Point is None or champ2Point is None:
                continue
            similarity = math.sqrt(champ1Point * champ2Point) / pointDict['sum']
            
            # (champ1 index, champ2 index, dot product) Not champion id!!)
            result.append((CHAMP_TO_INDEX[comb[0]],CHAMP_TO_INDEX[comb[1]],similarity))
            
        result = torch.tensor(result)
        return result
    
class Model(nn.Module):
    def __init__(self, nChampion, nEmbedding):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(nChampion, nEmbedding)
        
    def forward(self, champ1, champ2): # champ1 : (batchsize, 1)
        # print(champ1,champ2)
        # champ1Embedding = torch.unsqueeze(self.embedding(champ1),1)
        # champ2Embedding = torch.unsqueeze(self.embedding(champ2),2)
        # print(champ1Embedding)
        # print(champ2Embedding)
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
    
train_dataset = Dataset(summonerMasteryResult)
training_generator = torch.utils.data.DataLoader(train_dataset, shuffle=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

net = Model(161,15)#.cuda()
net.train()
optimizer = optim.Adam(net.parameters())
iteration = 0

# what I learn:
# TODO: add additional layer to network
# TODO: crawl more ids
# TODO: finish training code

for account in training_generator:
    # optimizer.zero_grad()
    champ1 = account[0,:,0].to(torch.int) # champ1 index list
    champ2 = account[0,:,1].to(torch.int) # champ2 index list
    gt_score = account[0,:,2] # champion dot product
    pred_score = net(champ1,champ2)
    print(pred_score)
    print(gt_score)
    
    # loss = F.L2_loss(pred_score, gt_score)
    # loss.backward()
    # optimizer.step()
    # iteration += 1
