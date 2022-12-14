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

def save_mastery(region, tier):
    REGION = region
    TIER = tier

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
    champion_info_response = requests.get(URL_CHAMPION_INFO)

    ID_TO_CHAMP = {}
    for champName in champion_info_response.json()['data'].keys():
        ID_TO_CHAMP[int(champion_info_response.json()['data'][champName]['key'])] = champName.lower()    

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

    # Fill table
    connId = dbEngine.connect()

    # Load summoner ids
    sel = select(summoner_id_table.c.id) 
    idList = connId.execute(sel)
    connId.close()

    # Create DB
    masteryEngine = create_engine(f'sqlite:///database/summoner_mastery_{TIER}.db'.format(), echo=False)

    connMastery = masteryEngine.connect()

    connMastery.execute(text("drop table if exists summoner_mastery_table;")) # Is this work?
    result = connMastery.execute(text("create table summoner_mastery_table (id NVARCHAR(63), sum INTEGER, aatrox INTEGER, ahri INTEGER, akali INTEGER, akshan INTEGER, alistar INTEGER, amumu INTEGER, anivia INTEGER, annie INTEGER, aphelios INTEGER, ashe INTEGER, aurelionsol INTEGER, azir INTEGER, bard INTEGER, belveth INTEGER, blitzcrank INTEGER, brand INTEGER, braum INTEGER, caitlyn INTEGER, camille INTEGER, cassiopeia INTEGER, chogath INTEGER, corki INTEGER, darius INTEGER, diana INTEGER, draven INTEGER, drmundo INTEGER, ekko INTEGER, elise INTEGER, evelynn INTEGER, ezreal INTEGER, gwen INTEGER, fiddlesticks INTEGER, fiora INTEGER, fizz INTEGER, galio INTEGER, gangplank INTEGER, garen INTEGER, gnar INTEGER, gragas INTEGER, graves INTEGER, hecarim INTEGER, heimerdinger INTEGER, illaoi INTEGER, irelia INTEGER, ivern INTEGER, janna INTEGER, jarvaniv INTEGER, jax INTEGER, jayce INTEGER, jhin INTEGER, jinx INTEGER, kaisa INTEGER, kalista INTEGER, karma INTEGER, karthus INTEGER, kassadin INTEGER, katarina INTEGER, kayle INTEGER, kayn INTEGER, kennen INTEGER, khazix INTEGER, kindred INTEGER, kled INTEGER, kogmaw INTEGER, leblanc INTEGER, leesin INTEGER, leona INTEGER, lillia INTEGER, lissandra INTEGER, lucian INTEGER, lulu INTEGER, lux INTEGER, malphite INTEGER, malzahar INTEGER, maokai INTEGER, masteryi INTEGER, missfortune INTEGER, monkeyking INTEGER, mordekaiser INTEGER, morgana INTEGER, nami INTEGER, nasus INTEGER, nautilus INTEGER, neeko INTEGER, nidalee INTEGER, nilah INTEGER, nocturne INTEGER, nunu INTEGER, olaf INTEGER, orianna INTEGER, ornn INTEGER, pantheon INTEGER, poppy INTEGER, pyke INTEGER, qiyana INTEGER, quinn INTEGER, rakan INTEGER, rammus INTEGER, reksai INTEGER, rell INTEGER, renata INTEGER, renekton INTEGER, rengar INTEGER, riven INTEGER, rumble INTEGER, ryze INTEGER, samira INTEGER, sejuani INTEGER, senna INTEGER, seraphine INTEGER, sett INTEGER, shaco INTEGER, shen INTEGER, shyvana INTEGER, singed INTEGER, sion INTEGER, sivir INTEGER, skarner INTEGER, sona INTEGER, soraka INTEGER, swain INTEGER, sylas INTEGER, syndra INTEGER, tahmkench INTEGER, taliyah INTEGER, talon INTEGER, taric INTEGER, teemo INTEGER, thresh INTEGER, tristana INTEGER, trundle INTEGER, tryndamere INTEGER, twistedfate INTEGER, twitch INTEGER, udyr INTEGER, urgot INTEGER, varus INTEGER, vayne INTEGER, veigar INTEGER, velkoz INTEGER, vex INTEGER, vi INTEGER, viego INTEGER, viktor INTEGER, vladimir INTEGER, volibear INTEGER, warwick INTEGER, xayah INTEGER, xerath INTEGER, xinzhao INTEGER, yasuo INTEGER, yone INTEGER, yorick INTEGER, yuumi INTEGER, zac INTEGER, zed INTEGER, zeri INTEGER, ziggs INTEGER, zilean INTEGER, zoe INTEGER, zyra INTEGER, primary key (id));"))

    meta.reflect(bind = masteryEngine)

    # Get mastery points
    URL_MASTERY_BY_ID = 'https://{region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/{summonerId}?api_key={apiKey}'

    apiCall = 0
    checkUniqueId = []

    for row in tqdm(idList):
        if row.id in checkUniqueId:
            logging.debug('Duplicate ID')
            continue
        summonerMasteryResponse = requests.get(URL_MASTERY_BY_ID.format(
                                            region = REGION,
                                            summonerId = row.id,
                                            apiKey = apikey))
        apiCall += 1
        checkUniqueId.append(row.id)
        summonerMasteryInfo = json.loads(summonerMasteryResponse.text)
        summonerMasteryDic = {}
        summonerMasteryDic['id'] = row.id
        pointSum = 0
        # Response exception
        try:
            for elem in summonerMasteryInfo:
                champId = elem['championId']
                champPoints = elem['championPoints']
                pointSum += champPoints
                summonerMasteryDic[ID_TO_CHAMP[champId]] = champPoints
        except:
            break
        summonerMasteryDic['sum'] = pointSum
        
        summonerMasteryDf = pd.DataFrame(summonerMasteryDic, index = [0])
        summonerMasteryDf.to_sql('summoner_mastery_table', connMastery, if_exists='append', index=False)
        if apiCall%99 == 0:
            time.sleep(120)

    sel = select(meta.tables['summoner_mastery_table']) # get all column if blank 
    result = connMastery.execute(sel)
    connMastery.close()