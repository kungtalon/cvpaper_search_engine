import logging
import re
import time
import os
import sys
import requests
import tarfile as tar
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
from pdfminer3.layout import LAParams
from pdfminer3.pdfparser import PDFParser
from pdfminer3.pdfdocument import PDFDocument
from pdfminer3.pdfpage import PDFPage
from pdfminer3.converter import PDFPageAggregator
from pdfminer3 import layout
from io import StringIO, BytesIO


logging.basicConfig(filename='./sections.log', filemode="a", level=logging.ERROR, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class DownloadError(Exception):
    def __init__(self):
        super(DownloadError, self).__init__()

def log(msg, type='error'):
    if type == 'error':
        logging.error(msg)
    elif type == 'info':
        logging.info(msg)
    elif type == 'debug':
        logging.debug(msg)

def download(url, download_path, timeout=180):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; hu-HU; rv:1.7.8) Gecko/20050511 Firefox/1.0.4'}
    response = requests.get(url, headers=headers, stream=True, timeout=timeout)
    if response.status_code == 200:
        with open(download_path, 'wb') as f:
            f.write(response.raw.read())
    else:
        log(f"Error requests for {url}!!!")
        raise DownloadError


def parse_tex_from_tar(file_path):
    with tar.open(file_path) as f:
        tex_files = [t for t in f.getnames() if t.endswith('.tex')]
        res = ''
        for tex in tex_files:
            try:
                tmp = f.extractfile(tex).read()
            except:
                raise tar.ReadError
            try:
                res += tmp.decode('utf-8')
            except:
                try:
                    res += tmp.decode('cp1250').encode().decode('utf-8')
                except:
                    res += tmp.decode('utf-8', errors='ignore')
    return res

def parse_sections_from_tex(raw_tex: str):
    sections = re.findall('\\\\section{([a-zA-Z0-9 ]*)}', raw_tex)
    subsections = re.findall('\\\\subsection{([a-zA-Z0-9 ]*)}', raw_tex)
    return [r.lower() for r in sections + subsections]

def delete_path(file_path):
    try:
        os.remove(file_path)
        log(f'Succeeded deleting file {file_path} ...', type='info')
    except:
        log(f'Failed to delete file {file_path} ...')

def extract_text_from_pdf(file_path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device =  TextConverter(rsrcmgr, retstr, codec='utf-8', laparams=laparams)
    with open(file_path, 'rb') as fp:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ''
        maxpages = 0
        caching = True
        pagenos = set()

        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()

    retstr.close()
    return text

def parse_sections_from_pdf(file_path):
    '''
    Use PDFMiner to extract texts from PDF
    Use RegEx(re) to select titles of subsections
    '''
    try:
        raw = extract_text_from_pdf(file_path)
    except:
        raw = ''
    try:
        candidates = parse_title_sections(file_path)
    except:
        candidates = ['']
    reg_res = [can for can in candidates if re.match(r'^(?:[1-9]\.)+ +[0-9a-zA-Zα-ωΑ-Ω -=]+[a-zA-Zα-ωΑ-Ω][0-9a-zA-Zα-ωΑ-Ω -=]+\n$', can)]
    subsections = [re.sub('^[0-9\.]+', '', r.strip()).strip().lower() for r in reg_res]
    subsections = [re.sub('\(cid:173\)', '-', r) for r in subsections]
    subsections = [re.sub('\(cid:[0-9]+\)', ' ', r) for r in subsections]
    return raw, subsections

def createPDFDoc(fpath):
    fp = open(fpath, 'rb')
    parser = PDFParser(fp)
    document = PDFDocument(parser, password='')
    # Check if the document allows text extraction. If not, abort.
    if not document.is_extractable:
        raise "Not extractable"
    else:
        return fp, document


def createDeviceInterpreter():
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    return device, interpreter


def parse_obj(objs):
    '''
    Iterate through all elements in pdf
    Check the font of each line
    '''
    res = []
    for obj in objs:
        if isinstance(obj, layout.LTTextBox):
            for o in obj._objs:
                if isinstance(o,layout.LTTextLine):
                    text=o.get_text()
                    if text.strip():
                        if text[0].isnumeric():
                            if isinstance(o._objs[0], layout.LTChar):
                                if o._objs[0].fontname.endswith('Medi') or o._objs[0].fontname.endswith('Bold'):
                                    res.append(text)
        # if it's a container, recurse
        elif isinstance(obj, layout.LTFigure):
            res.extend(parse_obj(obj._objs))
        else:
            pass
    return res

def parse_obj_debug(objs):
    res = []
    for obj in objs:
        if isinstance(obj, layout.LTTextBox):
            for o in obj._objs:
                if isinstance(o,layout.LTTextLine):
                    text=o.get_text()
                    if text.strip():
                        for c in o._objs:
                            if isinstance(c, layout.LTChar):
                                res.append((text, c.fontname))
                                break
        # if it's a container, recurse
        elif isinstance(obj, layout.LTFigure):
            res.extend(parse_obj(obj._objs))
        else:
            pass
    return res

def parse_title_sections(path, debug=False):
    '''
    We can extract possible text by checking the font of each line
    Only xxx-Bold and xxx-Medi are valid.
    '''
    res = []
    fp, document = createPDFDoc(path)
    device,interpreter = createDeviceInterpreter()
    pages = PDFPage.create_pages(document)
    for page in pages:
        interpreter.process_page(page)
        layout = device.get_result()
        if debug:
            res.extend(parse_obj_debug(layout._objs))
        else:
            res.extend(parse_obj(layout._objs))

    fp.close()
    return res
