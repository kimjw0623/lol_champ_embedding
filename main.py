import os
import logging
import csv

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

ID_TO_CHAMP = {}
for champName in champion_info_response.json()['data'].keys():
    ID_TO_CHAMP[int(champion_info_response.json()['data'][champName]['key'])] = champName.lower()

###########################################################################

summonerNameList = ['hide on bush','unpause','칼과 창 방패']

with open('id_list.csv', mode='r', encoding='utf-8-sig') as inp:
    reader = csv.reader(inp)
    for rows in reader:
        summonerNameList.append(rows[1])

URL_ID_BY_NAME = 'https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{userName}?api_key={apiKey}'

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
connId = dbEngine.connect()
connId.close()
summoner_id_table.drop(dbEngine)

# Fill table
connId = dbEngine.connect()

for summonerName in summonerNameList:
    summonerInfoResponse = requests.get(URL_ID_BY_NAME.format(
                                        region = REGION,
                                        userName = summonerName,
                                        apiKey = apikey))

    summonerInfo = json.loads(summonerInfoResponse.text)
    print(summonerInfo)
    summonerInfoDf = pd.DataFrame(summonerInfo, index = [0])[['id', 'accountId','puuid']]
    summonerInfoDf.to_sql('summoner_ids', connId, if_exists='append')

sel = select(summoner_id_table.c.id) # get all column if blank 
get_id_result = connId.execute(sel)

# for row in get_id_result:
#     print(row.id)
#     print('adsf')

######################################################

# Create DB
masteryEngine = create_engine('sqlite:///summoner_mastery.db', echo=False)

connMastery = masteryEngine.connect()

connMastery.execute(text("drop table if exists summoner_mastery_table;")) # Is this work?
result = connMastery.execute(text("create table summoner_mastery_table (id NVARCHAR(63), sum INTEGER, aatrox INTEGER, ahri INTEGER, akali INTEGER, akshan INTEGER, alistar INTEGER, amumu INTEGER, anivia INTEGER, annie INTEGER, aphelios INTEGER, ashe INTEGER, aurelionsol INTEGER, azir INTEGER, bard INTEGER, belveth INTEGER, blitzcrank INTEGER, brand INTEGER, braum INTEGER, caitlyn INTEGER, camille INTEGER, cassiopeia INTEGER, chogath INTEGER, corki INTEGER, darius INTEGER, diana INTEGER, draven INTEGER, drmundo INTEGER, ekko INTEGER, elise INTEGER, evelynn INTEGER, ezreal INTEGER, gwen INTEGER, fiddlesticks INTEGER, fiora INTEGER, fizz INTEGER, galio INTEGER, gangplank INTEGER, garen INTEGER, gnar INTEGER, gragas INTEGER, graves INTEGER, hecarim INTEGER, heimerdinger INTEGER, illaoi INTEGER, irelia INTEGER, ivern INTEGER, janna INTEGER, jarvaniv INTEGER, jax INTEGER, jayce INTEGER, jhin INTEGER, jinx INTEGER, kaisa INTEGER, kalista INTEGER, karma INTEGER, karthus INTEGER, kassadin INTEGER, katarina INTEGER, kayle INTEGER, kayn INTEGER, kennen INTEGER, khazix INTEGER, kindred INTEGER, kled INTEGER, kogmaw INTEGER, leblanc INTEGER, leesin INTEGER, leona INTEGER, lillia INTEGER, lissandra INTEGER, lucian INTEGER, lulu INTEGER, lux INTEGER, malphite INTEGER, malzahar INTEGER, maokai INTEGER, masteryi INTEGER, missfortune INTEGER, monkeyking INTEGER, mordekaiser INTEGER, morgana INTEGER, nami INTEGER, nasus INTEGER, nautilus INTEGER, neeko INTEGER, nidalee INTEGER, nilah INTEGER, nocturne INTEGER, nunu INTEGER, olaf INTEGER, orianna INTEGER, ornn INTEGER, pantheon INTEGER, poppy INTEGER, pyke INTEGER, qiyana INTEGER, quinn INTEGER, rakan INTEGER, rammus INTEGER, reksai INTEGER, rell INTEGER, renata INTEGER, renekton INTEGER, rengar INTEGER, riven INTEGER, rumble INTEGER, ryze INTEGER, samira INTEGER, sejuani INTEGER, senna INTEGER, seraphine INTEGER, sett INTEGER, shaco INTEGER, shen INTEGER, shyvana INTEGER, singed INTEGER, sion INTEGER, sivir INTEGER, skarner INTEGER, sona INTEGER, soraka INTEGER, swain INTEGER, sylas INTEGER, syndra INTEGER, tahmkench INTEGER, taliyah INTEGER, talon INTEGER, taric INTEGER, teemo INTEGER, thresh INTEGER, tristana INTEGER, trundle INTEGER, tryndamere INTEGER, twistedfate INTEGER, twitch INTEGER, udyr INTEGER, urgot INTEGER, varus INTEGER, vayne INTEGER, veigar INTEGER, velkoz INTEGER, vex INTEGER, vi INTEGER, viego INTEGER, viktor INTEGER, vladimir INTEGER, volibear INTEGER, warwick INTEGER, xayah INTEGER, xerath INTEGER, xinzhao INTEGER, yasuo INTEGER, yone INTEGER, yorick INTEGER, yuumi INTEGER, zac INTEGER, zed INTEGER, zeri INTEGER, ziggs INTEGER, zilean INTEGER, zoe INTEGER, zyra INTEGER, primary key (id));"))

meta.reflect(bind = masteryEngine)

# Get mastery points
URL_MASTERY_BY_ID = 'https://{region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/{summonerId}?api_key={apiKey}'

idList = connId.execute(sel)
for row in idList:
    summonerMasteryResponse = requests.get(URL_MASTERY_BY_ID.format(
                                        region = REGION,
                                        summonerId = row.id,
                                        apiKey = apikey))
    summonerMasteryInfo = json.loads(summonerMasteryResponse.text)
    summonerMasteryDic = {}
    summonerMasteryDic['id'] = row.id
    pointSum = 0
    for elem in summonerMasteryInfo:
        champId = elem['championId']
        champPoints = elem['championPoints']
        pointSum += champPoints
        summonerMasteryDic[ID_TO_CHAMP[champId]] = champPoints
        
    summonerMasteryDic['sum'] = pointSum
    summonerMasteryDf = pd.DataFrame(summonerMasteryDic, index = [0])
    summonerMasteryDf.to_sql('summoner_mastery_table', connMastery, if_exists='append', index=False)

sel = select(meta.tables['summoner_mastery_table']) # get all column if blank 
result = connMastery.execute(sel)

for row in result:
    print(row)