import os
import logging
import csv
import time
from tqdm import tqdm

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
   
# REGION = 'KR'
# TIER = 'master'

def save_id(region, tier):
    REGION = region
    TIER = tier
    REMOVE_ID_TABLE = True
    GET_NEW_ID = True
    GET_NEW_MASTERY_POINT = False

    # URL-Endpoints
    URL_CHAMPION_INFO = 'http://ddragon.leagueoflegends.com/cdn/12.14.1/data/en_US/champion.json'
    URL_MASTERY_POINTS = 'https://{region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/{summonerId}?api_key={apiKey}'
    URL_ID_BY_NAME = 'https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{userName}?api_key={apiKey}'

    # Get api key
    load_dotenv(verbose=True)
    apikey = os.getenv('RIOT_API_KEY')

    if apikey is None:
        logging.error('No api key in .env file')
        raise AssertionError('No api key in .env file')
    else:
        logging.debug('Get api key successfully')

    # Get champion info
    champion_info_response = requests.get(URL_CHAMPION_INFO)

    ID_TO_CHAMP = {}
    for champName in champion_info_response.json()['data'].keys():
        ID_TO_CHAMP[int(champion_info_response.json()['data'][champName]['key'])] = champName.lower()

    # Get summoner name list
    summonerNameList = []
    with open(f'summoner_name/id_list_{TIER}.csv', mode='r', encoding='utf-8-sig') as inp:
        reader = csv.reader(inp)
        for rows in reader:
            summonerNameList.append(rows[1])

    # Create DB
    dbEngine = create_engine(f'sqlite:///database/summoner_ids_{TIER}.db', echo=False)

    # Create table
    meta = MetaData()
    summoner_id_table = Table(
        'summoner_ids', meta,
        Column('id', String, primary_key=True),
        Column('accountId', String),
        Column('puuid', String),
    )

    # Drop table if already exists
    if REMOVE_ID_TABLE:
        meta.create_all(dbEngine)
        connId = dbEngine.connect()
        connId.close()
        summoner_id_table.drop(dbEngine)

    # Fill table
    connId = dbEngine.connect()

    # Save summoner id info
    if GET_NEW_ID:
        apiCall = 0
        for summonerName in tqdm(summonerNameList):
            apiCall += 1
            summonerInfoResponse = requests.get(URL_ID_BY_NAME.format(region = REGION,
                                                                    userName = summonerName,
                                                                    apiKey = apikey))
            summonerInfo = json.loads(summonerInfoResponse.text)
            try:
                summonerInfo['id']
            except:
                continue
            summonerInfoDf = pd.DataFrame(summonerInfo, index = [0])[['id', 'accountId','puuid']]
            summonerInfoDf.to_sql('summoner_ids', connId, if_exists='append')
            # api call limitation (100 / 2min.)
            if apiCall%90 == 0:
                print('sleep')
                time.sleep(130)
            
    sel = select(summoner_id_table.c.id) # get all column if blank 
    get_id_result = connId.execute(sel)
    connId.close()