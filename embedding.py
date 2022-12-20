import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import requests
import itertools
import math
import time
from einops import rearrange, reduce, repeat
from PIL import Image
from io import BytesIO 
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn import manifold 

from sklearn.decomposition import PCA

# SQL and ORM
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData
from sqlalchemy.sql import select

from option import args
import get_id
import get_masteryinfo

MIN_MASTERY_POINT = 10
MIN_SUM_MASTERY_POINT = 500000

# Get champion list w/ id
URL_CHAMPION_INFO = 'http://ddragon.leagueoflegends.com/cdn/12.14.1/data/en_US/champion.json'
champion_info_response = requests.get(URL_CHAMPION_INFO)

ID_TO_CHAMP = {}
for champName in champion_info_response.json()['data'].keys():
    ID_TO_CHAMP[int(champion_info_response.json()['data'][champName]['key'])] = champName.lower()

CHAMP_TO_INDEX = {}
CHAMP_TO_INDEX = dict(zip(ID_TO_CHAMP.values(),range(162)))
CHAMP_NUM = len(CHAMP_TO_INDEX)

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
            if champ1Point is None:
                champ1Point = MIN_MASTERY_POINT
            if champ2Point is None:
                champ2Point = MIN_MASTERY_POINT
            similarity = math.sqrt(champ1Point * champ2Point) / pointDict['sum'] # normalize point?
            # (champ1 index, champ2 index, dot product): Not champion id!!
            result.append((CHAMP_TO_INDEX[comb[0]],CHAMP_TO_INDEX[comb[1]],similarity))
            
        result = torch.tensor(result)
        return result
    
    # Select champion pair with weight
    def _champion_choice(self, champPointDict):
        # Give weight according to the mastery point
        pointList = [10 if x==None else pow(int(x),0.5) for x in list(champPointDict.values())[2:]]
        pointList = np.array(pointList)/np.ndarray.sum(np.array(pointList))
        assert(len(pointList) == len(list(ID_TO_CHAMP.values())))

        # Select 20 champions
        champWeightedList = list(np.random.choice(np.array(list(ID_TO_CHAMP.values())), p=pointList, size=20, replace=False))
        champComb = list(itertools.combinations(champWeightedList, 2)) # list of tuple (champName, champName) <- string
        
        return champComb
    
class Model(nn.Module):
    def __init__(self, champ_num, emb_num):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(champ_num, emb_num)
        
    def forward(self, champ1, champ2): # champ1 : (batchsize, 1)
        dotResult = torch.bmm(torch.unsqueeze(self.embedding(champ1),1),
                              torch.unsqueeze(self.embedding(champ2),2))
        return dotResult

def main():
    # Setting & Hyperparameters 
    get_data = args.get_data
    is_train = args.is_train

    region = args.region
    tier = args.tier
    embedding_num = args.embed_num
    total_epoch = args.tot_epoch
    batch_size = args.batch_size

    # Get data using API. If not, use demo data
    if get_data:
        get_id.save_id(region,tier)
        get_masteryinfo.save_mastery(region,tier)

    # Get summoner's mastery points    
    masteryEngine = create_engine('sqlite:///database/summoner_mastery_master.db', echo=False)
    connMastery = masteryEngine.connect()
    meta = MetaData()
    meta.reflect(bind = masteryEngine)
    sel = select(meta.tables['summoner_mastery_table']) # get all column if blank 
    masteryResult = connMastery.execute(sel)

    summonerMasteryResult = []
    for elem in masteryResult:
        if elem['sum'] > MIN_SUM_MASTERY_POINT:
            summonerMasteryResult.append(dict(elem))
    time.sleep(1)

    batch_size = 64

    # Training step
    train_dataset = Dataset(summonerMasteryResult)
    training_generator = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f'Torch device: {device}')

    net = Model(CHAMP_NUM,embedding_num).to(device)
    net.train()
    optimizer = optim.Adam(net.parameters())
    cur_iteration = 0
    total_iter = int(total_epoch*train_dataset.__len__() / batch_size)

    if is_train:
        for epoch in range(total_epoch):
            for account in training_generator: # account: (B, N, 3)
                optimizer.zero_grad()
                account = rearrange(account, 'B C A -> (B C) A').to(device)
                champ1 = account[:,0].to(torch.int) # Champoin id
                champ2 = account[:,1].to(torch.int) # Champoin id
                gt_score = account[:,2] # Dot product of mastery point 
                pred_score = net(champ1,champ2).view(-1,)
                loss = F.mse_loss(pred_score, gt_score) # Get batch avg loss
                cur_iteration += 1
                print(f'Iter: {cur_iteration:06d} / {total_iter:06d} | Loss: {loss.item():.6f}')
                loss.backward()
                optimizer.step()
        # Save model
        torch.save({'model_state_dict': net.state_dict()}, 'models/embedding.pt')
    else:
        # Load model
        net.load_state_dict(torch.load('models/embedding.pt')['model_state_dict'])
        net.eval()

    # Embedding vector visualization
    for param in net.parameters():
        champ_embedding = param.cpu().detach().numpy()
        
    champ_embedding = champ_embedding / np.linalg.norm(champ_embedding, axis = 1).reshape(-1, 1)

    for i in range(CHAMP_NUM):
        champ_vector = (np.expand_dims(champ_embedding[i],0) @ champ_embedding.T)[0]
        champ_dict = dict(zip(ID_TO_CHAMP.values(), champ_vector))
        champ_dict = dict(sorted(champ_dict.items(), key=lambda item: item[1], reverse=True))

    # Championbinder
    CHAMPION_BINDER_URL = 'http://ddragon.leagueoflegends.com/cdn/12.14.1/img/champion/'
    ICON_ZOOM = 0.03

    champion_binder = list()
    champ_binder_list = list(champion_info_response.json()['data'].keys())
    for champion in tqdm(champ_binder_list, total = len(champ_binder_list)):
        while True:
            bindAntwort = requests.get(f'{CHAMPION_BINDER_URL}{champion}.png')
            if bindAntwort.status_code == 200:
                break
            else:
                time.sleep(0.1)
        champion_binder.append(Image.open(BytesIO(bindAntwort.content)))
                
    # Dimensionality reduction
    reducing_method = 'tsne'
    if reducing_method == 'tsne':
        tsne = manifold.TSNE(n_components = 2)
        reduced_matrix = tsne.fit_transform(champ_embedding)
    elif reducing_method == 'mds':
        mds = manifold.MDS(n_components = 2)
        reduced_matrix = mds.fit_transform(champ_embedding)
    else:
        pca = PCA(n_components=2)
        reduced_matrix = pca.fit_transform(champ_embedding)

    tsne2dDf = pd.DataFrame(reduced_matrix, columns = ['X1', 'X2'])
    tsne2dDf['Champion'] = champ_binder_list

    # Matplotlib
    fig = plt.figure(figsize=(3, 2), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(tsne2dDf['X1'], tsne2dDf['X2'], color = 'white')
    ax.axis('off')
    for i in range(0, len(champ_binder_list)):
        bindBox = OffsetImage(champion_binder[i], zoom = ICON_ZOOM)
        ax.add_artist(AnnotationBbox(bindBox, tsne2dDf.iloc[i][['X1', 'X2']].values, frameon = False))

    fig.savefig(f"results/champion_clustering_{reducing_method}_{region}_{tier}.png", transparent = False, bbox_inches = 'tight', pad_inches = 0, dpi = 1000)
    plt.close()

if __name__ == '__main__':
    main()