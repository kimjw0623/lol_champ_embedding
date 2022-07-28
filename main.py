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



URL_ID_BY_NAME = 'https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{userName}?api_key={apiKey}'

summonerInfoResponse = requests.get(URL_ID_BY_NAME.format(
                                    region = REGION,
                                    userName = INITIAL_ACCOUNT,
                                    apiKey = apikey))

summonerInfo = json.loads(summonerInfoResponse.text)
summonerInfoDf = pd.DataFrame(summonerInfo, index = [0])[['id', 'accountId','puuid']]
summonerInfoDfDic = summonerInfoDf.loc[0] #dictionary?
print(summonerInfoDfDic['id'])
# Create DB
dbEngine = create_engine('sqlite:///summoner_ids.db', echo=True)

# Create table
meta = MetaData()
#meta.__table__.drop(dbEngine)
summoner_id_table = Table(
    'summoner_ids', meta,
    Column('id', String, primary_key=True),
    Column('accountId', String),
    Column('puuid', String),
)
meta.create_all(dbEngine)

ins = summoner_id_table.insert().values(id = summonerInfoDfDic['id'],
                                        accountId = summonerInfoDfDic['accountId'],
                                        puuid = summonerInfoDfDic['puuid'],)
conn = dbEngine.connect()
result = conn.execute(ins)

sel = summoner_id_table.select()
result = conn.execute(sel)

for row in result:
    print(row)
# with dbEngine.connect() as conn:
#     result = conn.execute(
#         insert(summoner_id_table).values()
#     )
#     conn.commit()

# URL_ID_ZU_NAME = 'https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{userName}?api_key={apiKey}'

# DB_NAME = 'E_LOL_CHAMPS_EMBEDDING'
# DB_SCHEMA = 'LOL_CHAMPS_EMBEDDING'

# DB_TABELLE_IDS = 'SUMMONER_IDS'

# initialerAccountAntwort = requests.get(URL_ID_ZU_NAME.format(
#                                         region = REGION,
#                                         userName = INITIAL_ACCOUNT,
#                                         apiKey = apikey))

# initialerAccount = json.loads(initialerAccountAntwort.text)
# initialerAccountDf = pd.DataFrame(initialerAccount, index = [0])[['id', 'accountId']]

# initialerAccountDf.to_sql('{}_{}'.format(DB_TABELLE_IDS, REGION.upper()), 
#                           dbEngine, 
#                           # schema = '{}.{}'.format(DB_NAME, DB_SCHEMA), 
#                           index = False, 
#                           if_exists = 'append')

# print(initialerAccountDf)

# a = pd.read_sql('select count(*) as N_ACCOUNTS from {}_{}'.format(DB_TABELLE_IDS, REGION.upper()),dbEngine)['N_ACCOUNTS'].iloc[0]
# print(a)
# b = pd.read_sql('select id, accountId from {}_{}'.format(DB_TABELLE_IDS, REGION.upper()),dbEngine)
# print(b)