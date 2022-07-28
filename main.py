import os
import logging

import requests
import urllib3

from dotenv import load_dotenv

import pandas as pd
import numpy as np
import json

from sqlalchemy import create_engine

# 1. Define DB & ORM
# 2. Get summoner id list w/ crawling
# 3. 

REGION = 'KR'
INITIAL_ACCOUNT = 'unpause'

# URL-Endpoints
URL_CHAMPION_INFO = 'http://ddragon.leagueoflegends.com/cdn/12.14.1/data/en_US/champion.json'
URL_MASTERY_POINTS = 'https://{region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/{summonerId}?api_key={apiKey}'


DSN = 'E_LOCAL'

# Get api key
load_dotenv(verbose=True)
apikey = os.getenv('RIOT_API_KEY')

if apikey is None:
    logging.error('No api key in .env file')
    raise AssertionError('No api key in .env file')
else:
    logging.debug('Get api key successfully')
    
champion_info_response = requests.get(URL_CHAMPION_INFO)

print(champion_info_response.json()['data']['Zoe'])

dbEngine = create_engine('sqlite:///:memory:', echo=True)


URL_ID_ZU_NAME = 'https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{userName}?api_key={apiKey}'

DB_NAME = 'E_LOL_CHAMPS_EMBEDDING'
DB_SCHEMA = 'LOL_CHAMPS_EMBEDDING'

DB_TABELLE_IDS = 'SUMMONER_IDS'

initialerAccountAntwort = requests.get(URL_ID_ZU_NAME.format(
                                        region = REGION,
                                        userName = INITIAL_ACCOUNT,
                                        apiKey = apikey))

initialerAccount = json.loads(initialerAccountAntwort.text)
initialerAccountDf = pd.DataFrame(initialerAccount, index = [0])[['id', 'accountId']]

initialerAccountDf.to_sql('{}_{}'.format(DB_TABELLE_IDS, REGION.upper()), 
                          dbEngine, 
                          # schema = '{}.{}'.format(DB_NAME, DB_SCHEMA), 
                          index = False, 
                          if_exists = 'append')

print(initialerAccountDf)

a = pd.read_sql('select count(*) as N_ACCOUNTS from {}_{}'.format(DB_TABELLE_IDS, REGION.upper()),dbEngine)['N_ACCOUNTS'].iloc[0]
print(a)
b = pd.read_sql('select id, accountId from {}_{}'.format(DB_TABELLE_IDS, REGION.upper()),dbEngine)
print(b)