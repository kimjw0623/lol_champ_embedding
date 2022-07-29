import os
import logging

import requests
import urllib3

from dotenv import load_dotenv

import pandas as pd
import numpy as np
import json

# SQL and ORM
import sqlite3
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData
from sqlalchemy.sql import select
from sqlalchemy import text

# 1. Define DB & ORM
# 2. Get summoner id list w/ crawling
# 3. 

REGION = 'KR'
INITIAL_ACCOUNT = 'unpause'

# URL-Endpoints
URL_CHAMPION_INFO = 'http://ddragon.leagueoflegends.com/cdn/12.14.1/data/en_US/champion.json'
URL_MASTERY_POINTS = 'https://{region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/{summonerId}?api_key={apiKey}'

# Get api key
load_dotenv(verbose=True)
apikey = os.getenv('RIOT_API_KEY')

if apikey is None:
    logging.error('No api key in .env file')
    raise AssertionError('No api key in .env file')
else:
    logging.debug('Get api key successfully')

# Get champion info  
# TODO: get champion id  
champion_info_response = requests.get(URL_CHAMPION_INFO)
print(champion_info_response.json()['data']['Zoe'])

###########################################################################

summonerNameList = ['hide on bush']#,'unpause','칼과 창 방패']

URL_ID_BY_NAME = 'https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{userName}?api_key={apiKey}'

summonerInfoDicList = []
for summonerName in summonerNameList:
    summonerInfoResponse = requests.get(URL_ID_BY_NAME.format(
                                        region = REGION,
                                        userName = summonerName,
                                        apiKey = apikey))

    summonerInfo = json.loads(summonerInfoResponse.text)
    summonerInfoDic = {}
    summonerInfoDic['id'] = summonerInfo['id']
    summonerInfoDic['accountId'] = summonerInfo['accountId']
    summonerInfoDic['puuid'] = summonerInfo['puuid']
    summonerInfoDicList.append(summonerInfoDic)
    
    summonerInfoDf = pd.DataFrame(summonerInfo, index = [0])[['id', 'accountId','puuid']]
    # summonerInfoDfDic = summonerInfoDf.loc[0] #dictionary: {'id': 'daf32fsdafa','accountId': 'asdafsafc'}

# Create DB
dbEngine = create_engine('sqlite:///summoner_ids.db', echo=False)

# Create table
meta = MetaData()
summoner_id_table = Table(
    'summoner_ids', meta,
    Column('id', String, primary_key=True),
    Column('accountId', String),
    Column('puuid', String),
)

# Drop existing table
meta.create_all(dbEngine)
conn = dbEngine.connect()
conn.close()
summoner_id_table.drop(dbEngine)

# Fill table
# meta.create_all(dbEngine)
conn = dbEngine.connect()
summonerInfoDf.to_sql('summoner_ids',conn)
#result = conn.execute(summoner_id_table.insert(), summonerInfoDicList) # using 

sel = select(summoner_id_table.c.id) # get all column if blank 
get_id_result = conn.execute(sel)

for row in get_id_result:
    print(row.id)
    print('adsf')

######################################################

# Create DB
masteryEngine = create_engine('sqlite:///summoner_mastery.db', echo=False)

# meta.create_all(masteryEngine)
# conn = masteryEngine.connect()
# conn.close()
# meta.tables['summoner_mastery_table'].drop(masteryEngine)

conn = masteryEngine.connect()

conn.execute(text("drop table if exists summoner_mastery_table;")) # Is this work?
result = conn.execute(text("create table summoner_mastery_table (id NVARCHAR(63), sum INTEGER, aatrox INTEGER, ahri INTEGER, akali INTEGER, alistar INTEGER, amumu INTEGER, anivia INTEGER, annie INTEGER, aphelios INTEGER, ashe INTEGER, aurelionsol INTEGER, azir INTEGER, bard INTEGER, blitzcrank INTEGER, brand INTEGER, braum INTEGER, caitlyn INTEGER, camille INTEGER, cassiopeia INTEGER, chogath INTEGER, corki INTEGER, darius INTEGER, diana INTEGER, draven INTEGER, drmundo INTEGER, ekko INTEGER, elise INTEGER, evelynn INTEGER, ezreal INTEGER, fiddlesticks INTEGER, fiora INTEGER, fizz INTEGER, galio INTEGER, gangplank INTEGER, garen INTEGER, gnar INTEGER, gragas INTEGER, graves INTEGER, hecarim INTEGER, heimerdinger INTEGER, illaoi INTEGER, irelia INTEGER, ivern INTEGER, janna INTEGER, jarvaniv INTEGER, jax INTEGER, jayce INTEGER, jhin INTEGER, jinx INTEGER, kaisa INTEGER, kalista INTEGER, karma INTEGER, karthus INTEGER, kassadin INTEGER, katarina INTEGER, kayle INTEGER, kayn INTEGER, kennen INTEGER, khazix INTEGER, kindred INTEGER, kled INTEGER, kogmaw INTEGER, leblanc INTEGER, leesin INTEGER, leona INTEGER, lillia INTEGER, lissandra INTEGER, lucian INTEGER, lulu INTEGER, lux INTEGER, malphite INTEGER, malzahar INTEGER, maokai INTEGER, masteryi INTEGER, missfortune INTEGER, monkeyking INTEGER, mordekaiser INTEGER, morgana INTEGER, nami INTEGER, nasus INTEGER, nautilus INTEGER, neeko INTEGER, nidalee INTEGER, nocturne INTEGER, nunu INTEGER, olaf INTEGER, orianna INTEGER, ornn INTEGER, pantheon INTEGER, poppy INTEGER, pyke INTEGER, qiyana INTEGER, quinn INTEGER, rakan INTEGER, rammus INTEGER, reksai INTEGER, renekton INTEGER, rengar INTEGER, riven INTEGER, rumble INTEGER, ryze INTEGER, sejuani INTEGER, senna INTEGER, sett INTEGER, shaco INTEGER, shen INTEGER, shyvana INTEGER, singed INTEGER, sion INTEGER, sivir INTEGER, skarner INTEGER, sona INTEGER, soraka INTEGER, swain INTEGER, sylas INTEGER, syndra INTEGER, tahmkench INTEGER, taliyah INTEGER, talon INTEGER, taric INTEGER, teemo INTEGER, thresh INTEGER, tristana INTEGER, trundle INTEGER, tryndamere INTEGER, twistedfate INTEGER, twitch INTEGER, udyr INTEGER, urgot INTEGER, varus INTEGER, vayne INTEGER, veigar INTEGER, velkoz INTEGER, vi INTEGER, viktor INTEGER, vladimir INTEGER, volibear INTEGER, warwick INTEGER, xayah INTEGER, xerath INTEGER, xinzhao INTEGER, yasuo INTEGER, yone INTEGER, yorick INTEGER, yuumi INTEGER, zac INTEGER, zed INTEGER, ziggs INTEGER, zilean INTEGER, zoe INTEGER, zyra INTEGER, primary key (id));"))
# conn.commit()

# Get mastery points
URL_MASTERY_BY_ID = 'https://{region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/{summonerId}?api_key={apiKey}'
ID_TO_CHAMP = {1:'Ahri'}

summonerMasteryDicList = []
for row in get_id_result:
    print(row.id)
    summonerId = row.id
    summonerMasteryResponse = requests.get(URL_MASTERY_BY_ID.format(
                                        region = REGION,
                                        summonerId = summonerId,
                                        apiKey = apikey))
    summonerMasteryInfo = json.loads(summonerMasteryResponse.text)
    summonerMasteryDic = {}
    summonerMasteryDic['id'] = summonerId
    summonerMasteryDic['sum'] = 1234
    summonerMasteryDic['aatrox'] = 442
    print(summonerMasteryDic)
    summonerMasteryDf = pd.DataFrame(summonerMasteryDic, index = [0])
    summonerMasteryDicList.append(summonerMasteryDic)
    
    # for champPointDic in summonerMasteryInfo:
    #     champId = int(champPointDic['championId'])
    #     champPoints = champPointDic['championPoints']
    #     ID_TO_CHAMP[champId] = champPoints
    
    # summonerMasteryDic = {champ:champ for champPointDic in summonerMasteryInfo}
    # Generate dictionary for a single row and 

# Fill table
# meta.create_all(masteryEngine)
meta.reflect(bind = masteryEngine)
# conn = masteryEngine.connect()
# result = conn.execute(meta.tables['summoner_mastery_table'].insert(), summonerMasteryDicList)

summonerMasteryDf.to_sql('summoner_mastery_table', conn, if_exists='append', index=False)

sel = select(meta.tables['summoner_mastery_table']) # get all column if blank 
result = conn.execute(sel)

for row in result:
    print(row)