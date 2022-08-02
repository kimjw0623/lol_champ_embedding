import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import csv

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome('./chromedriver', options=options)
#regex = re.compile('[^a-zA-Z]')

summoners_id_list = []
with open('id_list.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    spamwriter = csv.writer(csvfile)
    idx = 0
    for page_num in range(3000,3200):
        info_url = f'https://www.op.gg/leaderboards/tier?page={page_num}&region=kr'
        driver.get(info_url) 
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        time.sleep(3)
        # Get top5 summoners
        if page_num == 1:
            top5 = soup.select_one('div', class_='css-1j84o5i ei93w703')
            top1 = top5.find('a',class_='name')
            spamwriter.writerow([idx,top1.text])
            #summoners_id_list.append(top1.text) 
            idx += 1

            top5 = top5.find_all('span', class_ = 'name')
            for i in range(4):
                spamwriter.writerow([idx,top5[i].text])
                #summoners_id_list.append(top5[i].text) 
                idx += 1

        others = soup.select_one('table', class_= 'css-168jvq5 exo2f213')
        rows = others.select('tr',class_='css-1kk0pwf e1g3wlsd9')
        for i in range(len(rows)-1):
            spamwriter.writerow([idx,rows[i+1].find('strong').text])
            #summoners_id_list.append(rows[i+1].find('strong').text)
            idx += 1

# with open('id_list_10000.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
#     spamwriter = csv.writer(csvfile)
#     for i in range(len(summoners_id_list)):
#         spamwriter.writerow([i,summoners_id_list[i]])

driver.quit()