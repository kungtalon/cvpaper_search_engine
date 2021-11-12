import json
import time
import pandas as pd
import os
from os.path import join as pjoin
from tqdm import tqdm
from utils import log
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By

FIELDS = ['conference', 'year', 'workshop', 'title', 'authors', 'pdf_link', 'arxiv_link', 'supp_link', 'abstract', 'key_words', 'subsections']
DETAIL_FIELDS = ['title', 'authors', 'pdf_link', 'arxiv_link', 'supp_link', 'abstract', 'key_words', 'subsections']

class PaperDataParser:
    def __init__(self, target_json_path, driver, output_dir):
        self.fields = FIELDS
        self.targets = json.load(open(target_json_path, 'r'))
        self.driver = driver
        self.output_dir = output_dir
        self.exists = set([f.replace('.xls', '') for f in os.listdir(self.output_dir) if f.endswith('xls')])
        self.data = self.parse_all()

    def parse_all(self):
        papers_data = []
        for conference, target_urls in tqdm(self.targets.items()):
            if conference in self.exists:
                continue
            if isinstance(target_urls, str):
                papers = self.parse_conference(target_urls)  # all paper data in current conference with only DETAIL_FIELDS
            else:
                papers = []
                for target_url in target_urls:
                    papers.extend(self.parse_conference(target_url))
            conference_name = conference[:4]
            year = conference[4:8]
            workshop = conference.split('_')[-1] if 'workshop' in conference else ''
            all_fields = [(conference_name, year, workshop) + paper_fields for paper_fields in papers]
            self.save_dataframe(all_fields, conference)
            papers_data.extend(all_fields)
        return papers_data

    def save_dataframe(self, data, target_name):
        target_name = target_name.replace('/', ' ')
        df_data = pd.DataFrame(data, columns=FIELDS)
        df_data.to_excel(pjoin(self.output_dir, f'{target_name}.xls'))
        df_data.to_pickle(pjoin(self.output_dir, f'{target_name}.pkl'))

    def parse_conference(self, url):
        '''
        input : the url with a list of papers in a conference, e.g. https://openaccess.thecvf.com/ICCV2015
        return : a list of lists, each sub-list conresponds to a paper with DETAIL_FIELDS data
        '''
        conference_papers = []
        self.driver.get(url)
        paper_urls = [obj.get_attribute('href') for obj in self.driver.find_elements_by_xpath("//div[@id='content']/dl/dt[@class='ptitle']/a")]
        for paper_url in paper_urls:
            paper_detail = self.parse_paper_detail(paper_url)
            if paper_detail is not None:
                conference_papers.append(paper_detail)
        return conference_papers

    def parse_paper_detail(self, paper_url):
        '''
        input : the detail page for a paper, e.g. https://openaccess.thecvf.com/content_iccv_2015/html/Malinowski_Ask_Your_Neurons_ICCV_2015_paper.html
        return : the DETAIL_FIELDS data for this paper, or None if errors occur
        '''
        try:
            self.driver.get(paper_url)
            title = self.driver.find_element_by_xpath("//div[@id='papertitle']").text
            authors = self.driver.find_element_by_xpath("//div[@id='authors']/b/i").text
            links = self.driver.find_elements_by_xpath("//div[@id='content']/dl/dd/a")
            pdf_link = [obj.get_attribute('href') for obj in links if obj.text == 'pdf'][0]
            supp_link_list = [obj.get_attribute('href') for obj in links if obj.text == 'supp']
            supp_link = supp_link_list[0] if supp_link_list else ''
            arxiv_link_list = [obj.get_attribute('href') for obj in links if obj.text == 'arXiv']
            arxiv_link = arxiv_link_list[0] if arxiv_link_list else ''
            abstract = self.driver.find_element_by_xpath("//div[@id='abstract']").text
            key_words = ''
            subsections = ''
            scope = locals()
            return tuple(eval(field, scope) for field in DETAIL_FIELDS)
        except Exception as ex:
            log(f'Failed to parse paper {paper_url}. Msg: {ex.msg}')
            return None


def main():
    options = webdriver.FirefoxOptions()
    options.add_argument('--headless')
    driver = webdriver.Firefox(executable_path='/home/kungtalon/drivers/geckodriver', options=options)
    parser = PaperDataParser('./target_new.json', driver, './')
    
if __name__ == '__main__':
    main()