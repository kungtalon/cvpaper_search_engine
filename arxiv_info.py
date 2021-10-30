from genericpath import exists
from sre_constants import SUCCESS
import tarfile
from selenium.common import exceptions
import feedparser
import re
import os
import time
import json
import requests
import random
import pandas as pd
import requests
import warnings
import Levenshtein
import utils
from utils import log
from tqdm import tqdm
from unidecode import unidecode
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from urllib.request import urlopen
from string import punctuation

warnings.simplefilter('ignore')

class ProxyDriver:
    def __init__(self, proxies_file):
        self.cur_proxy_idx = 0 
        if not proxies_file:
            log('No proxy.')
            self.proxies = []
            self.create_driver()
        else:
            with open(proxies_file, 'r') as init_proxies:
                self.proxies = [(r['protocols'][0] + '://' + r['ip'] + ':' + r['port'], r['protocols'][0], r['ip'], r['port']) for r in json.load(init_proxies)]
            # self.validate_ip()
            self.create_driver(self.proxies[self.cur_proxy_idx][0])

    def validate_ip(self):
        valid_proxies = []
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Safari/537.36'}
        for proxy in self.proxies:
            if proxy[1] in ['socks5', 'socks4']:
                trial_proxy = {'http':proxy[0], 'https':proxy[0]}
            else:
                trial_proxy = {'http':f'http://{proxy[2]}:{proxy[3]}', 'https':f'https://{proxy[2]}:{proxy[3]}'}
            try:
                request = requests.get('http://icanhazip.com/', headers=header, proxies=trial_proxy, timeout=4)
                if request.status_code == 200:
                    valid_proxies.append(proxy)
                else:
                    log(f'{proxy[0]} is not valid, deleted...')
            except requests.exceptions.RequestException:
                log(f'{proxy[0]} is not valid, deleted...')
        self.proxies = valid_proxies
        log(f'Number of remaining proxies: ' + str(len(self.proxies)))

    def update_driver(self):
        self.cur_proxy_idx += 1
        if self.cur_proxy_idx == len(self.proxies):
            raise IndexError('Ran out of proxies!')
        self.driver.close()
        self.create_driver(self.proxies[self.cur_proxy_idx][0])

    def create_driver(self, proxy=None):
        options = webdriver.FirefoxOptions()
        options.add_argument('--headless')
        if proxy:
            options.add_argument(f'--proxy-server={proxy}')
        self.driver = webdriver.Firefox(executable_path='/home/kungtalon/drivers/geckodriver', options=options)
    
    def get(self, url):
        self.driver.get(url)

    def find_elements_by_xpath(self, path):
        return self.driver.find_elements_by_xpath(path)

    def actions(self):
        return ActionChains(self.driver)

    def execute_script(self, scr):
        self.driver.execute_script(scr)

    def refresh(self):
        self.driver.refresh()

    def back(self):
        self.driver.back()

class ArxivSearcher:
    def __init__(self, driver=None) -> None:
        self.driver = driver
        self.web_action = self.driver.actions()

    def search_by_title(self, title, authors):
        query = '+'.join(re.sub(f'[{punctuation}]', ' ', title).split(' '))
        # use unicode to avoid non-english letters causing encoding issues
        author_set = set(unidecode(authors).split(', '))
        # results = self.parse_title_authors_feed(self.arxiv_api_search(unidecode(query)))
        results = self.arxiv_selenium_search(query)
        for res in results:
            if Levenshtein.distance(res[0].lower(), title.lower()) <= 2:
                return res[-1], 2
        
        results.sort(key = lambda x: len(author_set.symmetric_difference(x[1])))
        top = results[0]
        author_diff = len(author_set.symmetric_difference(top[1]))
        if author_diff == 0:
            return top[-1], 2
        elif author_diff <= 2:
            return top[-1], 1
        else:
            return top[-1], 0

    def arxiv_api_search(self, query: str):
        base_url = 'http://export.arxiv.org/api/query?'
        api_query = f'search_query=ti:{query}&start=0&max_results=5'
        return urlopen(base_url + api_query).read()
        # response = requests.get(base_url + api_query, proxies=self.proxies).text
        # return response

    def arxiv_selenium_search(self, query):
        success = False
        base_url = 'https://arxiv.org/search/?query='
        search_query = f'{query}&searchtype=title&abstracts=hide&size=25&order='
        while not success:
            try:
                self.driver.get(base_url+search_query)
                # add some human-like actions
                if random.random() < 0.5:
                    time.sleep(3 + random.random() * 5)
                else:
                    time.sleep(2 + min(random.normalvariate(3, 1), 6))
                if random.random() < 0.1:
                    self.driver.refresh()
                    time.sleep(1 + random.random() * 2)
                if random.random() < 0.01:
                    time.sleep(110 + random.random() * 10)
                titles_elems = self.driver.find_elements_by_xpath('//p[@class="title is-5 mathjax"]')
                authors_elems = self.driver.find_elements_by_xpath('//p[@class="authors"]')
                links_elems = self.driver.find_elements_by_xpath('//li[@class="arxiv-result"]/div[@class="is-marginless"]/p/a')
                titles = [t.text for t in titles_elems]
                authors = [set([unidecode(a.text) for a in au.find_elements_by_xpath('./a')]) for au in authors_elems]
                links = [l.get_attribute('href') for l in links_elems]
                if links_elems and random.random() < 0.414:
                    self.web_action.move_to_element(links_elems[0]).click().perform()
                    time.sleep(2 + random.random() * 2)
                    if random.random() < 0.632:
                        self.driver.back()
                        time.sleep(2)
                elif random.random() < 0.5:
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1.676 + random.random())
                success = True
            except WebDriverException as ex:
                if self.driver.proxies:
                    self.driver.update_driver()
                    self.web_action = self.driver.actions()
                else:
                    raise ex
        assert len(titles) == len(authors) == len(links)
        if len(titles):
            return list(zip(titles, authors, links))[:min(5, len(titles))]
        return [[' ', set([' ']), '']]

    def parse_title_authors_feed(self, response):
        feeds = feedparser.parse(response)
        result = []
        if feeds.entries:
            for entry in feeds.entries:
                title = entry.title
                authors = [unidecode(dic['name']) for dic in entry.authors]
                # use unicode to avoid non-english letters causing discrepancy when comparing authors
                url = entry.link
                result.append([title, set(authors), url])
        else:
            result.append(['', set(['']), ''])
        return result


def search_paper_on_arxiv(src_dir, des_dir):
    driver = ProxyDriver('')

    src_excels = [file for file in os.listdir(src_dir) if file.endswith('.xls')]
    des_excels = set([file for file in os.listdir(des_dir) if file.endswith('.xls')])
    searcher = ArxivSearcher(driver)
    for excel_name in src_excels:
        if excel_name[4:8] < '2015':
            continue
        if excel_name in des_excels:
            continue
        log('Processing ' + excel_name)
        data = pd.read_excel(src_dir + excel_name)
        arxiv_link_list = []
        arxiv_link_assured_list = []
        for title, authors, arxiv_link, assured in tqdm(data[['title', 'authors', 'arxiv_link', 'arxiv_link_assured']].values):
            if str(arxiv_link) not in ['nan', '', ' '] and assured == 2:
                arxiv_link_list.append(arxiv_link)
                arxiv_link_assured_list.append(2)
                continue
            res = searcher.search_by_title(title, authors)
            arxiv_link_list.append(res[0])
            arxiv_link_assured_list.append(res[1])
        data['arxiv_link'] = pd.Series(arxiv_link_list)
        data['arxiv_link_assured'] = pd.Series(arxiv_link_assured_list)
        data.to_excel(des_dir + excel_name)

def parse_content_from_arxiv(src_dir, des_dir, download_dir):
    os.makedirs(des_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    src_excels = [file for file in os.listdir(src_dir) if file.endswith('.xls')]
    des_excels = set([file for file in os.listdir(des_dir) if file.endswith('.xls')])
    for excel_name in src_excels:
        if excel_name[4:8] < '2015':
            continue
        if excel_name in des_excels:
            continue
        log('Processing ' + excel_name)
        data = pd.read_excel(src_dir + excel_name)
        
        all_tex_raw = []
        all_pdf_raw = []
        all_subsections = []
        for title, pdf_link, arxiv_link, assured in tqdm(data[['title', 'pdf_link', 'arxiv_link', 'arxiv_link_assured']].values):
            tex = ''
            pdf_raw = ''
            if str(arxiv_link) not in ['nan', '', ' '] and assured == 2:
                try:
                    time.sleep(random.random()*3 + 3)
                    arxiv_id = arxiv_link.split('/')[-1]
                    extra_data_link = f'https://arxiv.org/e-print/{arxiv_id}'
                    target_path = os.path.join(download_dir, arxiv_id + '.tar')
                    utils.download(extra_data_link, target_path)
                    tex = utils.parse_tex_from_tar(target_path)
                    subsections = utils.parse_sections_from_tex(tex)
                except tarfile.ReadError:
                    log(f'tar file open erro for {title}')
            if not tex:
                # no arxiv link found, we can parse text from pdf
                target_path = os.path.join(download_dir, title + '.pdf')
                utils.download(pdf_link, target_path)
                pdf_raw, subsections = utils.parse_sections_from_pdf(target_path)
            if not len(''.join(subsections)):
                log(f'unable to get contents for {title}')
            all_tex_raw.append(tex)
            all_pdf_raw.append(pdf_raw)
            all_subsections.append(';'.join(subsections))
            utils.delete_path(target_path)
        data['tex_raw'] = pd.Series(all_tex_raw)
        data['pdf_raw'] = pd.Series(all_pdf_raw)
        data['subsections'] = pd.Series(all_subsections)
        data.to_excel(des_dir + excel_name)

if __name__ == '__main__':
    # search_paper_on_arxiv('./newdata', './newdatafix')
    parse_content_from_arxiv('./newdatafix/', './alldata/', './downloads/')
