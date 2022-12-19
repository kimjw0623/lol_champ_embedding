from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import csv

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome('./chromedriver', options=options)

summoners_id_list = []

# Save name of summoner as .csv file
with open('id_list_silver4.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    spamwriter = csv.writer(csvfile)
    idx = 0
    for page_num in range(26000,26100):
        info_url = f'https://www.op.gg/leaderboards/tier?page={page_num}&region=kr'
        driver.get(info_url) 
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        time.sleep(3)
        
        others = soup.select_one('table', class_= 'css-168jvq5 exo2f213')
        rows = others.select('tr',class_='css-1kk0pwf e1g3wlsd9')
        for i in range(len(rows)-1):
            spamwriter.writerow([idx,rows[i+1].find('strong').text])
            idx += 1
            
driver.quit()