import sqlite3
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

engine = create_engine('sqlite:///asd.db', echo=True)
from sqlalchemy import MetaData

from sqlalchemy import Table, Column, Integer, String, MetaData
meta = MetaData()
members = Table(
    'members', meta,
    Column('id', Integer, primary_key=True),
    Column('name', String),
    Column('age', Integer),
)
meta.create_all(engine)

ins = members.insert().values(name = 'RM', age = 26)

conn = engine.connect()
result = conn.execute(ins)

sel = members.select()
result = conn.execute(sel)

for row in result:
    print(row)

#########################################################

meta.create_all(engine)

dbpath = 'summoner_id.db'

conn = sqlite3.connect(dbpath)
cur = conn.cursor()

script = """
DROP TABLE IF EXISTS summoner_id;

CREATE TABLE summoner_id(
id TEXT PRIMARY KEY,
accountId TEXT NOT NULL,
puuid TEXT
);
"""

cur.executescript(script)
conn.commit()

data = [('aaa','bbb','qqq')]

cur.executemany("INSERT INTO summoner_id(id,accountId,puuid) VALUES(?,?,?)",data)
conn.commit()

cur.execute("SELECT * FROM summoner_id;")
summoner_id_list = cur.fetchall()

print(summoner_id_list)

df = pd.read_sql_query("SELECT * FROM summoner_id", conn) 
print(df)
print(df['id'][0])
df['id'][0] = 'ccc'
#df.to_sql('summoner_id',conn)
#conn.commit()



#df = pd.read_sql_query("SELECT * FROM summoner_id", conn) 
#print(df)

conn.close()

# Insert data in .db file
# for li in boxItems:
#     title = li.find_element_by_class_name('proTit').text # li.find_element_by_css_selector('h5.proTit').text
#     price = li.find_element_by_class_name('proPrice').text.replace(',','').replace('원~','') # li.find_element_by_css_selector('.proPrice')
#     image = li.find_element_by_class_name('img').get_attribute('src')

#     sql_query = "INSERT INTO tour_crawl(title, price, image) values('{}',{},'{}')".format(title, price, image) # TEXT인 제목은 ''로 감싸주는 것에 유의
#     print('SQL Query :', sql_query[:90], "...")

#     cur.execute(sql_query)
#     conn.commit()