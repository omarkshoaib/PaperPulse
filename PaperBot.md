This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
file_examples/
  jurnals.csv
  papers.txt
PyPaperBot/
  __init__.py
  __main__.py
  Crossref.py
  Downloader.py
  HTMLparsers.py
  NetInfo.py
  Paper.py
  PapersFilters.py
  proxy.py
  Scholar.py
  Utils.py
.gitignore
LICENSE.txt
README.md
requirements.txt
setup.cfg
setup.py
```

# Files

## File: file_examples/jurnals.csv
````
journal_list;include_list
journal name 1;1
journal name 2;0
journal name 3;1
````

## File: file_examples/papers.txt
````
paper DOI 1
paper DOI 2
paper DOI 3
paper DOI 4
paper DOI 5
````

## File: PyPaperBot/__init__.py
````python
__version__= "1.4.1"
````

## File: PyPaperBot/__main__.py
````python
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import time
import requests
from .Paper import Paper
from .PapersFilters import filterJurnals, filter_min_date, similarStrings
from .Downloader import downloadPapers
from .Scholar import ScholarPapersInfo
from .Crossref import getPapersInfoFromDOIs
from .proxy import proxy
from .__init__ import __version__
from urllib.parse import urljoin

def checkVersion():
    try :
        print("PyPaperBot v" + __version__)
        response = requests.get('https://pypi.org/pypi/pypaperbot/json')
        latest_version = response.json()['info']['version']
        if latest_version != __version__:
            print("NEW VERSION AVAILABLE!\nUpdate with 'pip install PyPaperBot —upgrade' to get the latest features!\n")
    except :
        pass


def start(query, scholar_results, scholar_pages, dwn_dir, proxy, min_date=None, num_limit=None, num_limit_type=None,
          filter_jurnal_file=None, restrict=None, DOIs=None, SciHub_URL=None, chrome_version=None, cites=None,
          use_doi_as_filename=False, SciDB_URL=None, skip_words=None):

    if SciDB_URL is not None and "/scidb" not in SciDB_URL:
        SciDB_URL = urljoin(SciDB_URL, "/scidb/")

    to_download = []
    if DOIs is None:
        print("Query: {}".format(query))
        print("Cites: {}".format(cites))
        to_download = ScholarPapersInfo(query, scholar_pages, restrict, min_date, scholar_results, chrome_version, cites, skip_words)
    else:
        print("Downloading papers from DOIs\n")
        num = 1
        i = 0
        while i < len(DOIs):
            DOI = DOIs[i]
            print("Searching paper {} of {} with DOI {}".format(num, len(DOIs), DOI))
            papersInfo = getPapersInfoFromDOIs(DOI, restrict)
            papersInfo.use_doi_as_filename = use_doi_as_filename
            to_download.append(papersInfo)

            num += 1
            i += 1

    if restrict != 0 and to_download:
        if filter_jurnal_file is not None:
            to_download = filterJurnals(to_download, filter_jurnal_file)

        if min_date is not None:
            to_download = filter_min_date(to_download, min_date)

        if num_limit_type is not None and num_limit_type == 0:
            to_download.sort(key=lambda x: int(x.year) if x.year is not None else 0, reverse=True)

        if num_limit_type is not None and num_limit_type == 1:
            to_download.sort(key=lambda x: int(x.cites_num) if x.cites_num is not None else 0, reverse=True)

        downloadPapers(to_download, dwn_dir, num_limit, SciHub_URL, SciDB_URL)

    Paper.generateReport(to_download, dwn_dir + "result.csv")
    Paper.generateBibtex(to_download, dwn_dir + "bibtex.bib")


def main():
    print(
        """PyPaperBot is a Python tool for downloading scientific papers using Google Scholar, Crossref and SciHub.
        -Join the telegram channel to stay updated --> https://t.me/pypaperbotdatawizards <--
        -If you like this project, you can share a cup of coffee at --> https://www.paypal.com/paypalme/ferru97 <-- :)\n""")
    time.sleep(4)
    parser = argparse.ArgumentParser(
        description='PyPaperBot is python tool to search and dwonload scientific papers using Google Scholar, Crossref and SciHub')
    parser.add_argument('--query', type=str, default=None,
                        help='Query to make on Google Scholar or Google Scholar page link')
    parser.add_argument('--skip-words', type=str, default=None,
                        help='List of comma separated works. Papers from Scholar containing this words on title or summary will be skipped')
    parser.add_argument('--cites', type=str, default=None,
                        help='Paper ID (from scholar address bar when you search citations) if you want get only citations of that paper')
    parser.add_argument('--doi', type=str, default=None,
                        help='DOI of the paper to download (this option uses only SciHub to download)')
    parser.add_argument('--doi-file', type=str, default=None,
                        help='File .txt containing the list of paper\'s DOIs to download')
    parser.add_argument('--scholar-pages', type=str,
                        help='If given in %%d format, the number of pages to download from the beginning. '
                             'If given in %%d-%%d format, the range of pages (starting from 1) to download (the end is included). '
                             'Each page has a maximum of 10 papers (required for --query)')
    parser.add_argument('--dwn-dir', type=str, help='Directory path in which to save the results')
    parser.add_argument('--min-year', default=None, type=int, help='Minimal publication year of the paper to download')
    parser.add_argument('--max-dwn-year', default=None, type=int,
                        help='Maximum number of papers to download sorted by year')
    parser.add_argument('--max-dwn-cites', default=None, type=int,
                        help='Maximum number of papers to download sorted by number of citations')
    parser.add_argument('--journal-filter', default=None, type=str,
                        help='CSV file path of the journal filter (More info on github)')
    parser.add_argument('--restrict', default=None, type=int, choices=[0, 1],
                        help='0:Download only Bibtex - 1:Down load only papers PDF')
    parser.add_argument('--scihub-mirror', default=None, type=str,
                        help='Mirror for downloading papers from sci-hub. If not set, it is selected automatically')
    parser.add_argument('--annas-archive-mirror', default=None, type=str,
                        help='Mirror for downloading papers from Annas Archive (SciDB). If not set, https://annas-archive.se is used')
    parser.add_argument('--scholar-results', default=10, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help='Downloads the first x results for each scholar page(default/max=10)')
    parser.add_argument('--proxy', nargs='+', default=[],
                        help='Use proxychains, provide a seperated list of proxies to use.Please specify the argument al the end')
    parser.add_argument('--single-proxy', type=str, default=None,
                        help='Use a single proxy. Recommended if using --proxy gives errors')
    parser.add_argument('--selenium-chrome-version', type=int, default=None,
                        help='First three digits of the chrome version installed on your machine. If provided, selenium will be used for scholar search. It helps avoid bot detection but chrome must be installed.')
    parser.add_argument('--use-doi-as-filename', action='store_true', default=False,
                        help='Use DOIs as output file names')
    args = parser.parse_args()

    if args.single_proxy is not None:
        os.environ['http_proxy'] = args.single_proxy
        os.environ['HTTP_PROXY'] = args.single_proxy
        os.environ['https_proxy'] = args.single_proxy
        os.environ['HTTPS_PROXY'] = args.single_proxy
        print("Using proxy: ", args.single_proxy)
    else:
        pchain = []
        pchain = args.proxy
        proxy(pchain)

    if args.query is None and args.doi_file is None and args.doi is None and args.cites is None:
        print("Error, provide at least one of the following arguments: --query, --file, or --cites")
        sys.exit()

    if (args.query is not None and args.doi_file is not None) or (args.query is not None and args.doi is not None) or (
            args.doi is not None and args.doi_file is not None):
        print("Error: Only one option between '--query', '--doi-file' and '--doi' can be used")
        sys.exit()

    if args.dwn_dir is None:
        print("Error, provide the directory path in which to save the results")
        sys.exit()

    if args.scholar_results != 10 and args.scholar_pages != 1:
        print("Scholar results best applied along with --scholar-pages=1")

    dwn_dir = args.dwn_dir.replace('\\', '/')
    if dwn_dir[-1] != '/':
        dwn_dir += "/"
    if not os.path.exists(dwn_dir):
        os.makedirs(dwn_dir, exist_ok=True)

    if args.max_dwn_year is not None and args.max_dwn_cites is not None:
        print("Error: Only one option between '--max-dwn-year' and '--max-dwn-cites' can be used ")
        sys.exit()

    if args.query is not None or args.cites is not None:
        if args.scholar_pages:
            try:
                split = args.scholar_pages.split('-')
                if len(split) == 1:
                    scholar_pages = range(1, int(split[0]) + 1)
                elif len(split) == 2:
                    start_page, end_page = [int(x) for x in split]
                    scholar_pages = range(start_page, end_page + 1)
                else:
                    raise ValueError
            except Exception:
                print(
                    r"Error: Invalid format for --scholar-pages option. Expected: %d or %d-%d, got: " + args.scholar_pages)
                sys.exit()
        else:
            print("Error: with --query provide also --scholar-pages")
            sys.exit()
    else:
        scholar_pages = 0

    DOIs = None
    if args.doi_file is not None:
        DOIs = []
        f = args.doi_file.replace('\\', '/')
        with open(f) as file_in:
            for line in file_in:
                if line[-1] == '\n':
                    DOIs.append(line[:-1])
                else:
                    DOIs.append(line)

    if args.doi is not None:
        DOIs = [args.doi]

    max_dwn = None
    max_dwn_type = None
    if args.max_dwn_year is not None:
        max_dwn = args.max_dwn_year
        max_dwn_type = 0
    if args.max_dwn_cites is not None:
        max_dwn = args.max_dwn_cites
        max_dwn_type = 1


    start(args.query, args.scholar_results, scholar_pages, dwn_dir, proxy, args.min_year , max_dwn, max_dwn_type ,
          args.journal_filter, args.restrict, DOIs, args.scihub_mirror, args.selenium_chrome_version, args.cites,
          args.use_doi_as_filename, args.annas_archive_mirror, args.skip_words)

if __name__ == "__main__":
    checkVersion()
    main()
    print(
        """\nWork completed!
        -Join the telegram channel to stay updated --> https://t.me/pypaperbotdatawizards <--
        -If you like this project, you can share a cup of coffee at --> https://www.paypal.com/paypalme/ferru97 <-- :)\n""")
````

## File: PyPaperBot/Crossref.py
````python
from crossref_commons.iteration import iterate_publications_as_json
from crossref_commons.retrieval import get_entity
from crossref_commons.types import EntityType, OutputType
from .PapersFilters import similarStrings
from .Paper import Paper
import requests
import time
import random


def getBibtex(DOI):
    try:
        url_bibtex = "http://api.crossref.org/works/" + DOI + "/transform/application/x-bibtex"
        x = requests.get(url_bibtex)
        if x.status_code == 404:
            return ""
        return str(x.text)
    except Exception as e:
        print(e)
        return ""


def getPapersInfoFromDOIs(DOI, restrict):
    paper_found = Paper()
    paper_found.DOI = DOI

    try:
        paper = get_entity(DOI, EntityType.PUBLICATION, OutputType.JSON)
        if paper is not None and len(paper) > 0:
            if "title" in paper:
                paper_found.title = paper["title"][0]
            if "short-container-title" in paper and len(paper["short-container-title"]) > 0:
                paper_found.jurnal = paper["short-container-title"][0]

            if restrict is None or restrict != 1:
                paper_found.setBibtex(getBibtex(paper_found.DOI))
    except:
        print("Paper not found " + DOI)

    return paper_found


# Get paper information from Crossref and return a list of Paper
def getPapersInfo(papers, scholar_search_link, restrict, scholar_results):
    papers_return = []
    num = 1
    for paper in papers:
        # while num <= scholar_results:
        title = paper['title']
        queries = {'query.bibliographic': title.lower(), 'sort': 'relevance',
                   "select": "DOI,title,deposited,author,short-container-title"}

        print("Searching paper {} of {} on Crossref...".format(num, len(papers)))
        num += 1

        found_timestamp = 0
        paper_found = Paper(title, paper['link'], scholar_search_link, paper['cites'], paper['link_pdf'], paper['year'],
                            paper['authors'])
        while True:
            try:
                for el in iterate_publications_as_json(max_results=30, queries=queries):

                    el_date = 0
                    if "deposited" in el and "timestamp" in el["deposited"]:
                        el_date = int(el["deposited"]["timestamp"])

                    if (paper_found.DOI is None or el_date > found_timestamp) and "title" in el and similarStrings(
                            title.lower(), el["title"][0].lower()) > 0.75:
                        found_timestamp = el_date

                        if "DOI" in el:
                            paper_found.DOI = el["DOI"].strip().lower()
                        if "short-container-title" in el and len(el["short-container-title"]) > 0:
                            paper_found.jurnal = el["short-container-title"][0]

                        if restrict is None or restrict != 1:
                            paper_found.setBibtex(getBibtex(paper_found.DOI))

                break
            except ConnectionError as e:
                print("Wait 10 seconds and try again...")
                time.sleep(10)

        papers_return.append(paper_found)

        time.sleep(random.randint(1, 10))

    return papers_return
````

## File: PyPaperBot/Downloader.py
````python
from os import path
import requests
import time
from .HTMLparsers import getSchiHubPDF, SciHubUrls
import random
from .NetInfo import NetInfo
from .Utils import URLjoin


def setSciHubUrl():
    print("Searching for a sci-hub mirror")
    r = requests.get(NetInfo.SciHub_URLs_repo, headers=NetInfo.HEADERS)
    links = SciHubUrls(r.text)

    for l in links:
        try:
            print("Trying with {}...".format(l))
            r = requests.get(l, headers=NetInfo.HEADERS)
            if r.status_code == 200:
                NetInfo.SciHub_URL = l
                break
        except:
            pass
    else:
        print(
            "\nNo working Sci-Hub instance found!\nIf in your country Sci-Hub is not available consider using a VPN or a proxy\nYou can use a specific mirror mirror with the --scihub-mirror argument")
        NetInfo.SciHub_URL = "https://sci-hub.st"


def getSaveDir(folder, fname):
    dir_ = path.join(folder, fname)
    n = 1
    while path.exists(dir_):
        n += 1
        dir_ = path.join(folder, f"({n}){fname}")

    return dir_


def saveFile(file_name, content, paper, dwn_source):
    f = open(file_name, 'wb')
    f.write(content)
    f.close()

    paper.downloaded = True
    paper.downloadedFrom = dwn_source


def downloadPapers(papers, dwnl_dir, num_limit, SciHub_URL=None, SciDB_URL=None):

    NetInfo.SciHub_URL = SciHub_URL
    if NetInfo.SciHub_URL is None:
        setSciHubUrl()
    if SciDB_URL is not None:
        NetInfo.SciDB_URL = SciDB_URL

    print("\nUsing Sci-Hub mirror {}".format(NetInfo.SciHub_URL))
    print("Using Sci-DB mirror {}".format(NetInfo.SciDB_URL))
    print("You can use --scidb-mirror and --scidb-mirror to specify your're desired mirror URL\n")

    num_downloaded = 0
    paper_number = 1
    paper_files = []
    for p in papers:
        if p.canBeDownloaded() and (num_limit is None or num_downloaded < num_limit):
            print("Download {} of {} -> {}".format(paper_number, len(papers), p.title))
            paper_number += 1

            pdf_dir = getSaveDir(dwnl_dir, p.getFileName())

            failed = 0
            url = ""
            while not p.downloaded and failed != 5:
                try:
                    dwn_source = 1  # 1 scidb - 2 scihub - 3 scholar
                    if failed == 0 and p.DOI is not None:
                        url = URLjoin(NetInfo.SciDB_URL, p.DOI)
                    if failed == 1 and p.DOI is not None:
                        url = URLjoin(NetInfo.SciHub_URL, p.DOI)
                        dwn_source = 2
                    if failed == 2 and p.scholar_link is not None:
                        url = URLjoin(NetInfo.SciHub_URL, p.scholar_link)
                    if failed == 3 and p.scholar_link is not None and p.scholar_link[-3:] == "pdf":
                        url = p.scholar_link
                        dwn_source = 3
                    if failed == 4 and p.pdf_link is not None:
                        url = p.pdf_link
                        dwn_source = 3

                    if url != "":
                        r = requests.get(url, headers=NetInfo.HEADERS)
                        content_type = r.headers.get('content-type')

                        if (dwn_source == 1 or dwn_source == 2) and 'application/pdf' not in content_type and "application/octet-stream" not in content_type:
                            time.sleep(random.randint(1, 4))

                            pdf_link = getSchiHubPDF(r.text)
                            if pdf_link is not None:
                                r = requests.get(pdf_link, headers=NetInfo.HEADERS)
                                content_type = r.headers.get('content-type')

                        if 'application/pdf' in content_type or "application/octet-stream" in content_type:
                            paper_files.append(saveFile(pdf_dir, r.content, p, dwn_source))
                except Exception:
                    pass

                failed += 1
````

## File: PyPaperBot/HTMLparsers.py
````python
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 11:59:42 2020

@author: Vito
"""
from bs4 import BeautifulSoup
import re


def schoolarParser(html):
    result = []
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.findAll("div", class_="gs_r gs_or gs_scl"):
        if not isBook(element):
            title = None
            link = None
            link_pdf = None
            cites = None
            year = None
            authors = None
            for h3 in element.findAll("h3", class_="gs_rt"):
                found = False
                for a in h3.findAll("a"):
                    if not found:
                        title = a.text
                        link = a.get("href")
                        found = True
            for a in element.findAll("a"):
                if "Cited by" in a.text:
                    cites = int(a.text[8:])
                if "[PDF]" in a.text:
                    link_pdf = a.get("href")
            for div in element.findAll("div", class_="gs_a"):
                try:
                    authors, source_and_year, source = div.text.replace('\u00A0', ' ').split(" - ")
                except ValueError:
                    continue

                if not authors.strip().endswith('\u2026'):
                    # There is no ellipsis at the end so we know the full list of authors
                    authors = authors.replace(', ', ';')
                else:
                    authors = None
                try:
                    year = int(source_and_year[-4:])
                except ValueError:
                    continue
                if not (1000 <= year <= 3000):
                    year = None
                else:
                    year = str(year)
            if title is not None:
                result.append({
                    'title': title,
                    'link': link,
                    'cites': cites,
                    'link_pdf': link_pdf,
                    'year': year,
                    'authors': authors})
    return result


def isBook(tag):
    result = False
    for span in tag.findAll("span", class_="gs_ct2"):
        if span.text == "[B]":
            result = True
    return result


def getSchiHubPDF(html):
    result = None
    soup = BeautifulSoup(html, "html.parser")

    iframe = soup.find(id='pdf') #scihub logic
    plugin = soup.find(id='plugin') #scihub logic
    download_scidb = soup.find("a", text=lambda text: text and "Download" in text, href=re.compile(r"\.pdf$")) #scidb logic
    embed_scihub = soup.find("embed") #scihub logic

    if iframe is not None:
        result = iframe.get("src")

    if plugin is not None and result is None:
        result = plugin.get("src")

    if result is not None and result[0] != "h":
        result = "https:" + result

    if download_scidb is not None and result is None:
        result = download_scidb.get("href")

    if embed_scihub is not None and result is None:
        result = embed_scihub.get("original-url")

    return result


def SciHubUrls(html):
    result = []
    soup = BeautifulSoup(html, "html.parser")

    for ul in soup.findAll("ul"):
        for a in ul.findAll("a"):
            link = a.get("href")
            if link.startswith("https://sci-hub.") or link.startswith("http://sci-hub."):
                result.append(link)

    return result
````

## File: PyPaperBot/NetInfo.py
````python
class NetInfo:
    SciHub_URL = None
    SciDB_URL = "https://annas-archive.se/scidb/"
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    SciHub_URLs_repo = "https://sci-hub.41610.org/"
````

## File: PyPaperBot/Paper.py
````python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 21:43:30 2020

@author: Vito
"""
import bibtexparser
import re
import pandas as pd
import urllib.parse


class Paper:


    def __init__(self,title=None, scholar_link=None, scholar_page=None, cites=None, link_pdf=None, year=None, authors=None):        
        self.title = title
        self.scholar_page = scholar_page
        self.scholar_link = scholar_link
        self.pdf_link = link_pdf
        self.year = year
        self.authors = authors

        self.jurnal = None
        self.cites_num = None
        self.bibtex = None
        self.DOI = None

        self.downloaded = False
        self.downloadedFrom = 0  # 1-SciHub 2-scholar
        
        self.use_doi_as_filename = False # if True, the filename will be the DOI

    def getFileName(self):
            try:
                if self.use_doi_as_filename:
                    return urllib.parse.quote(self.DOI, safe='') + ".pdf"
                else:
                    return re.sub(r'[^\w\-_. ]', '_', self.title) + ".pdf"
            except:
                return "none.pdf"

    def setBibtex(self, bibtex):
        x = bibtexparser.loads(bibtex, parser=None)
        x = x.entries

        self.bibtex = bibtex

        try:
            if "year" in x[0]:
                self.year = x[0]["year"]
            if 'author' in x[0]:
                self.authors = x[0]["author"]
            self.jurnal = x[0]["journal"].replace("\\", "") if "journal" in x[0] else None
            if self.jurnal is None:
                self.jurnal = x[0]["publisher"].replace("\\", "") if "publisher" in x[0] else None
        except:
            pass

    def canBeDownloaded(self):
        return self.DOI is not None or self.scholar_link is not None

    def generateReport(papers, path):
        # Define the column names
        columns = ["Name", "Scholar Link", "DOI", "Bibtex", "PDF Name",
                   "Year", "Scholar page", "Journal", "Downloaded",
                   "Downloaded from", "Authors"]

        # Prepare data to populate the DataFrame
        data = []
        for p in papers:
            pdf_name = p.getFileName() if p.downloaded else ""
            bibtex_found = p.bibtex is not None

            # Determine download source
            dwn_from = ""
            if p.downloadedFrom == 1:
                dwn_from = "SciDB"
            elif p.downloadedFrom == 2:
                dwn_from = "SciHub"
            elif p.downloadedFrom == 3:
                dwn_from = "Scholar"

            # Append row data as a dictionary
            data.append({
                "Name": p.title,
                "Scholar Link": p.scholar_link,
                "DOI": p.DOI,
                "Bibtex": bibtex_found,
                "PDF Name": pdf_name,
                "Year": p.year,
                "Scholar page": p.scholar_page,
                "Journal": p.jurnal,
                "Downloaded": p.downloaded,
                "Downloaded from": dwn_from,
                "Authors": p.authors
            })

        # Create a DataFrame and write to CSV
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path, index=False, encoding='utf-8')

    def generateBibtex(papers, path):
        content = ""
        for p in papers:
            if p.bibtex is not None:
                content += p.bibtex + "\n"

        relace_list = ["\ast", "*", "#"]
        for c in relace_list:
            content = content.replace(c, "")

        f = open(path, "w", encoding="latin-1", errors="ignore")
        f.write(str(content))
        f.close()
````

## File: PyPaperBot/PapersFilters.py
````python
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:41:29 2020

@author: Vito
"""
import pandas as pd
from difflib import SequenceMatcher


def similarStrings(a, b):
    return SequenceMatcher(None, a, b).ratio()


"""
Input
    papers: list of Paper
    csv_path: path of a csv containing the journals to include (consult the GitHub page for the csv format)
Output
    result: list of Paper published by the journals included in the csv
"""
def filterJurnals(papers,csv_path):
    result = []
    df = pd.read_csv(csv_path, sep=";")
    journal_list = list(df["journal_list"])
    include_list = list(df["include_list"])

    for p in papers:
        good = not (p.jurnal is not None and len(p.jurnal) > 0)
        if p.jurnal is not None:
            for jurnal, include in zip(journal_list, include_list):
                if include == 1 and similarStrings(p.jurnal, jurnal) >= 0.8:
                    good = True

        if good:
            result.append(p)

    return result


"""
Input
    papers: list of Paper
    min_year: minimal publication year accepted
Output
    result: list of Paper published since min_year
"""
def filter_min_date(list_papers,min_year):
    new_list = []

    for paper in list_papers:
        if paper.year is not None and int(paper.year) >= min_year:
            new_list.append(paper)

    return new_list
````

## File: PyPaperBot/proxy.py
````python
import socket
import pyChainedProxy as socks

def proxy(pchain):

    chain = pchain

    socks.setdefaultproxy()
    for hop in chain:
        socks.adddefaultproxy(*socks.parseproxy(hop))

    rawsocket = socket.socket
    socket.socket = socks.socksocket
````

## File: PyPaperBot/Scholar.py
````python
import time
import requests
import functools
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from .HTMLparsers import schoolarParser
from .Crossref import getPapersInfo
from .NetInfo import NetInfo


def waithIPchange():
    while True:
        inp = input('You have been blocked, try changing your IP or using a VPN. '
                    'Press Enter to continue downloading, or type "exit" to stop and exit....')
        if inp.strip().lower() == "exit":
            return False
        elif not inp.strip():
            print("Wait 30 seconds...")
            time.sleep(30)
            return True


def scholar_requests(scholar_pages, url, restrict, chrome_version, scholar_results=10):
    javascript_error = "Sorry, we can't verify that you're not a robot when JavaScript is turned off"
    to_download = []
    driver = None
    for i in scholar_pages:
        while True:
            res_url = url % (scholar_results * (i - 1))
            if chrome_version is not None:
                if driver is None:
                    print("Using Selenium driver")
                    options = Options()
                    options.add_argument('--headless')
                    driver = uc.Chrome(headless=True, use_subprocess=False, version_main=chrome_version)
                driver.get(res_url)
                html = driver.page_source
            else:
                html = requests.get(res_url, headers=NetInfo.HEADERS)
                html = html.text

            if javascript_error in html:
                is_continue = waithIPchange()
                if not is_continue:
                    return to_download
            else:
                break

        papers = schoolarParser(html)
        if len(papers) > scholar_results:
            papers = papers[0:scholar_results]

        print("\nGoogle Scholar page {} : {} papers found".format(i, scholar_results))

        if len(papers) > 0:
            papersInfo = getPapersInfo(papers, url, restrict, scholar_results)
            info_valids = functools.reduce(lambda a, b: a + 1 if b.DOI is not None else a, papersInfo, 0)
            print("Papers found on Crossref: {}/{}\n".format(info_valids, len(papers)))

            to_download.append(papersInfo)
        else:
            print("Paper not found...")

    return to_download


def parseSkipList(skip_words):
    skip_list = skip_words.split(",")
    print("Skipping results containing {}".format(skip_list))
    output_param = ""
    for skip_word in skip_list:
        skip_word = skip_word.strip()
        if " " in skip_word:
            output_param += '+-"' + skip_word + '"'
        else:
            output_param += '+-' + skip_word
    return output_param


def ScholarPapersInfo(query, scholar_pages, restrict, min_date=None, scholar_results=10, chrome_version=None, cites=None, skip_words=None):
    url = r"https://scholar.google.com/scholar?hl=en&as_vis=1&as_sdt=1,5&start=%d"
    if query:
        if len(query) > 7 and (query.startswith("http://") or query.startswith("https://")):
            url = query
        else:
            url += f"&q={query}"
        if skip_words:
            url += parseSkipList(skip_words)
            print(url)
    if cites:
        url += f"&cites={cites}"
    if min_date:
        url += f"&as_ylo={min_date}"

    to_download = scholar_requests(scholar_pages, url, restrict, chrome_version, scholar_results)

    return [item for sublist in to_download for item in sublist]
````

## File: PyPaperBot/Utils.py
````python
def URLjoin(*args):
    return "/".join(map(lambda x: str(x).rstrip('/'), args))
````

## File: .gitignore
````
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

#Visual studio settings
.vs
.vscode
````

## File: LICENSE.txt
````
MIT License

Copyright (c) 2020 Vito Ferrulli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
````

## File: README.md
````markdown
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.me/ferru97)

# NEWS: PyPaperBot development is back on track!
### Join the [Telegram](https://t.me/pypaperbotdatawizards) channel to stay updated, report bugs, or request custom data mining scripts.
---

# PyPaperBot

PyPaperBot is a Python tool for **downloading scientific papers and bibtex** using Google Scholar, Crossref, SciHub, and SciDB.
The tool tries to download papers from different sources such as PDF provided by Scholar, Scholar related links, and Scihub.
PyPaperbot is also able to download the **bibtex** of each paper.

## Features

- Download papers given a query
- Download papers given paper's DOIs
- Download papers given a Google Scholar link
- Generate Bibtex of the downloaded paper
- Filter downloaded paper by year, journal and citations number

## Installation

### For normal Users

Use `pip` to install from pypi:

```bash
pip install PyPaperBot
```

If on windows you get an error saying *error: Microsoft Visual C++ 14.0 is required..* try to install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/it/visual-cpp-build-tools/) or [Visual Studio](https://visualstudio.microsoft.com/it/downloads/)

### For Termux users

Since numpy cannot be directly installed....

```
pkg install wget
wget https://its-pointless.github.io/setup-pointless-repo.sh
pkg install numpy
export CFLAGS="-Wno-deprecated-declarations -Wno-unreachable-code"
pip install pandas
```

and

```
pip install PyPaperbot
```

## How to use

PyPaperBot arguments:

| Arguments                   | Description                                                                                                                                                                           | Type   |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| \-\-query                   | Query to make on Google Scholar or Google Scholar page link                                                                                                                           | string |
| \-\-skip-words              | List of comma separated words i.e. "word1,word2 word3,word4". Articles containing any of this word in the title or google scholar summary will be ignored                             | string |
| \-\-cites                   | Paper ID (from scholar address bar when you search cites) if you want get only citations of that paper                                                                                | string                              | string |
| \-\-doi                     | DOI of the paper to download (this option uses only SciHub to download)                                                                                                               | string |
| \-\-doi-file                | File .txt containing the list of paper's DOIs to download                                                                                                                             | string |
| \-\-scholar-pages           | Number or range of Google Scholar pages to inspect. Each page has a maximum of 10 papers                                                                                              | string |
| \-\-dwn-dir                 | Directory path in which to save the result                                                                                                                                            | string |
| \-\-min-year                | Minimal publication year of the paper to download                                                                                                                                     | int    |
| \-\-max-dwn-year            | Maximum number of papers to download sorted by year                                                                                                                                   | int    |
| \-\-max-dwn-cites           | Maximum number of papers to download sorted by number of citations                                                                                                                    | int    |
| \-\-journal-filter          | CSV file path of the journal filter (More info on github)                                                                                                                             | string |
| \-\-restrict                | 0:Download only Bibtex - 1:Download only papers PDF                                                                                                                                   | int    |
| \-\-scihub-mirror           | Mirror for downloading papers from sci-hub. If not set, it is selected automatically                                                                                                  | string |
| \-\-annas-archive-mirror    | Mirror for downloading papers from Annas Archive (SciDB). If not set, https://annas-archive.se is used                                                                                | string |
| \-\-scholar-results         | Number of scholar results to bedownloaded when \-\-scholar-pages=1                                                                                                                    | int    |
| \-\-proxy                   | Proxies to be used. Please specify the protocol to be used.                                                                                                                           | string |
| \-\-single-proxy            | Use a single proxy. Recommended if using --proxy gives errors.                                                                                                                        | string |
| \-\-selenium-chrome-version | First three digits of the chrome version installed on your machine. If provided, selenium will be used for scholar search. It helps avoid bot detection but chrome must be installed. | int    |
| \-\-use-doi-as-filename     | If provided, files are saved using the unique DOI as the filename rather than the default paper title                                                                                 | bool    |
| \-h                         | Shows the help                                                                                                                                                                        | --     |

### Note

You can use only one of the arguments in the following groups

- *\-\-query*, *\-\-doi-file*, and *\-\-doi* 
- *\-\-max-dwn-year* and *and max-dwn-cites*

One of the arguments *\-\-scholar-pages*, *\-\-query *, and* \-\-file* is mandatory
The arguments *\-\-scholar-pages* is mandatory when using *\-\-query *
The argument *\-\-dwn-dir* is mandatory

The argument *\-\-journal-filter*  require the path of a CSV containing a list of journal name paired with a boolean which indicates whether or not to consider that journal (0: don't consider /1: consider) [Example](https://github.com/ferru97/PyPaperBot/blob/master/file_examples/jurnals.csv)

The argument *\-\-doi-file*  require the path of a txt file containing the list of paper's DOIs to download organized with one DOI per line [Example](https://github.com/ferru97/PyPaperBot/blob/master/file_examples/papers.txt)

Use the --proxy argument at the end of all other arguments and specify the protocol to be used. See the examples to understand how to use the option.

## SciHub access

If access to SciHub is blocked in your country, consider using a free VPN service like [ProtonVPN](https://protonvpn.com/) 
Also, you can use proxy option above.

## Example

Download a maximum of 30 papers from the first 3 pages given a query and starting from 2018 using the mirror https://sci-hub.do:

```bash
python -m PyPaperBot --query="Machine learning" --scholar-pages=3  --min-year=2018 --dwn-dir="C:\User\example\papers" --scihub-mirror="https://sci-hub.do"
```

Download papers from pages 4 to 7 (7th included) given a query and skip words:

```bash
python -m PyPaperBot --query="Machine learning" --scholar-pages=4-7 --dwn-dir="C:\User\example\papers" --skip-words="ai,decision tree,bot"
```

Download a paper given the DOI:

```bash
python -m PyPaperBot --doi="10.0086/s41037-711-0132-1" --dwn-dir="C:\User\example\papers" -use-doi-as-filename`
```

Download papers given a file containing the DOIs:

```bash
python -m PyPaperBot --doi-file="C:\User\example\papers\file.txt" --dwn-dir="C:\User\example\papers"`
```

If it doesn't work, try to use *py* instead of *python* i.e.

```bash
py -m PyPaperBot --doi="10.0086/s41037-711-0132-1" --dwn-dir="C:\User\example\papers"`
```

Search papers that cite another (find ID in scholar address bar when you search citations):

```bash
python -m PyPaperBot --cites=3120460092236365926
```

Using proxy

```
python -m PyPaperBot --query=rheumatoid+arthritis --scholar-pages=1 --scholar-results=7 --dwn-dir=/download --proxy="http://1.1.1.1::8080,https://8.8.8.8::8080"
```
```
python -m PyPaperBot --query=rheumatoid+arthritis --scholar-pages=1 --scholar-results=7 --dwn-dir=/download -single-proxy=http://1.1.1.1::8080
```

In termux, you can directly use ```PyPaperBot``` followed by arguments...

## Contributions

Feel free to contribute to this project by proposing any change, fix, and enhancement on the **dev** branch

### To do

- Tests
- Code documentation
- General improvements

## Disclaimer

This application is for educational purposes only. I do not take responsibility for what you choose to do with this application.

## Donation

If you like this project, you can give me a cup of coffee :) 

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.me/ferru97)
````

## File: requirements.txt
````
astroid==3.3.5
attrs==24.2.0
beautifulsoup4==4.12.3
bibtexparser==1.4.2
certifi==2024.8.30
cffi==1.17.1
chardet==5.2.0
charset-normalizer==3.3.2
colorama==0.4.6
crossref-commons==0.0.7
dill==0.3.9
exceptiongroup==1.2.2
future==1.0.0
h11==0.14.0
HTMLParser==0.0.2
idna==2.10
isort==5.13.2
lazy-object-proxy==1.10.0
mccabe==0.7.0
numpy==2.1.2
outcome==1.3.0.post0
packaging==24.1
pandas==2.2.3
platformdirs==4.3.6
proxy.py==2.4.8
pyChainedProxy==1.3
pycparser==2.22
pylint==3.3.1
pyparsing==3.1.4
PySocks==1.7.1
python-dateutil==2.9.0.post0
python-dotenv==1.0.1
pytz==2024.2
ratelimit==2.2.1
requests==2.32.3
selenium==4.25.0
six==1.16.0
sniffio==1.3.1
sortedcontainers==2.4.0
soupsieve==2.6
toml==0.10.2
tomli==2.0.2
tomlkit==0.13.2
trio==0.26.2
trio-websocket==0.11.1
typing_extensions==4.12.2
tzdata==2024.2
undetected-chromedriver==3.5.5
urllib3==2.2.3
webdriver-manager==4.0.2
websocket-client==1.8.0
websockets==13.1
wrapt==1.16.0
wsproto==1.2.0
setuptools==75.2.0
````

## File: setup.cfg
````
[metadata]
description_file = README.md

[options.entry_points]
console_scripts =
    PyPaperBot = PyPaperBot.__main__:main
````

## File: setup.py
````python
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'PyPaperBot',        
  packages = setuptools.find_packages(),
  version = '1.4.1',
  license='MIT', 
  description = 'PyPaperBot is a Python tool for downloading scientific papers using Google Scholar, Crossref, and SciHub.',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'Vito Ferrulli',
  author_email = 'vitof970@gmail.com',
  url = 'https://github.com/ferru97/PyPaperBot',
  download_url = 'https://github.com/ferru97/PyPaperBot/archive/v1.4.1.tar.gz',
  keywords = ['download-papers','google-scholar', 'scihub', 'scholar', 'crossref', 'papers'],
  install_requires=[          
        'astroid>=2.4.2,<=2.5',
        'beautifulsoup4>=4.9.1',
        'bibtexparser>=1.2.0',
        'certifi>=2020.6.20',
        'chardet>=3.0.4',
        'colorama>=0.4.3',
        'crossref-commons>=0.0.7',
        'future>=0.18.2',
        'HTMLParser>=0.0.2',
        'idna>=2.10,<3',
        'isort>=5.4.2',
        'lazy-object-proxy>=1.4.3',
        'mccabe>=0.6.1',
        'numpy',
        'pandas',
        'pyChainedProxy>=1.1',
        'pylint>=2.6.0',
        'pyparsing>=2.4.7',
        'python-dateutil>=2.8.1',
        'pytz>=2020.1',
        'ratelimit>=2.2.1',
        'requests>=2.24.0',
        'six>=1.15.0',
        'soupsieve>=2.0.1',
        'toml>=0.10.1',
        'urllib3>=1.25.10',
        'wrapt>=1.12.1',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
  entry_points={
    'console_scripts': ["PyPaperBot=PyPaperBot.__main__:main"],
  },
)
````
