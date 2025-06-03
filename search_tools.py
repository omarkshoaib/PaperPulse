from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import time
import logging
from scholarly import scholarly, ProxyGenerator
from Bio import Entrez
import os
from datetime import datetime
from pydantic import Field
import re

# API Rate Limiting Configuration (matches research_pipeline.py)
API_CALL_DELAY = int(os.getenv("API_CALL_DELAY", "60"))  # Seconds between API calls

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PubMedSearchTool(BaseTool):
    name: str = "PubMedSearchTool"
    description: str = "A tool for searching and retrieving papers from PubMed"
    base_url: str = Field(default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")
    api_key: Optional[str] = Field(default=None)
    email: str = Field(default="omarkshoaib@gmail.com")
    max_results: int = Field(default=5)

    def _run(self, query: str) -> List[Dict[str, str]]:
        logger.info(f"Starting PubMed search for query: '{query}', max_results: {self.max_results}")
        
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key
        
        search_handle = Entrez.esearch(db="pubmed", term=query, retmax=str(self.max_results), idtype="acc")
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        paper_ids = search_results["IdList"]
        if not paper_ids:
            logger.info(f"No results found in PubMed for query: '{query}'")
            return []
        
        papers = []
        fetch_handle = Entrez.efetch(db="pubmed", id=paper_ids, rettype="xml")
        records = Entrez.read(fetch_handle)['PubmedArticle']
        fetch_handle.close()

        for record in records:
            try:
                article = record['MedlineCitation']['Article']
                title = article.get('ArticleTitle', '')
                
                authors_list = []
                if 'AuthorList' in article and article['AuthorList']:
                    for author_info in article['AuthorList']:
                        last_name = author_info.get('LastName', '')
                        fore_name = author_info.get('ForeName', '')
                        if last_name or fore_name:
                             authors_list.append(f"{fore_name} {last_name}".strip())
                authors_str = ", ".join(authors_list)

                year = ''
                if 'Journal' in article and 'JournalIssue' in article['Journal'] and 'PubDate' in article['Journal']['JournalIssue']:
                    pub_date = article['Journal']['JournalIssue']['PubDate']
                    year = pub_date.get('Year', '')
                    if not year and 'MedlineDate' in pub_date:
                        year_match = re.search(r'^(\d{4})', pub_date['MedlineDate'])
                        if year_match: year = year_match.group(1)

                original_abstract_text = ''
                if 'Abstract' in article and article['Abstract'] and article['Abstract'].get('AbstractText'):
                    abstract_parts = article['Abstract']['AbstractText']
                    if isinstance(abstract_parts, list):
                        original_abstract_text = " ".join(str(part) for part in abstract_parts)
                    else:
                        original_abstract_text = str(abstract_parts)
                
                pmid = record['MedlineCitation']['PMID']
                link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                paper_entry = {
                    'title': str(title).strip(),
                    'link': link,
                    'authors': authors_str.strip(),
                    'year': str(year).strip(),
                    'source': 'PubMed',
                    'original_abstract': original_abstract_text.strip(),
                }
                papers.append(paper_entry)
                logger.info(f"Scraped from PubMed: {paper_entry['title']}")
            except Exception as e:
                logger.error(f"Error processing PubMed record: {e}", exc_info=True)
            time.sleep(API_CALL_DELAY)
        
        logger.info(f"Total papers scraped from PubMed: {len(papers)}")
        return papers

class GoogleScholarSearchTool(BaseTool):
    name: str = "GoogleScholarSearchTool"
    description: str = "Search for papers on Google Scholar using the scholarly library."
    max_results: int = Field(default=5)
    
    def _run(self, query: str) -> List[Dict[str, str]]:
        logger.info(f"Starting Google Scholar scrape for query: '{query}', max_results: {self.max_results}")
        results: List[Dict[str, str]] = []
        
        try:
            search_iter = scholarly.search_pubs(query)
            count = 0
            for paper_obj in search_iter:
                if count >= self.max_results:
                    break
                try:
                    bib = paper_obj.get('bib', {})
                    title = bib.get('title', '')
                    if not title:
                        continue

                    authors_list = bib.get('author', [])
                    authors_str = ", ".join(authors_list) if isinstance(authors_list, list) else str(authors_list)
                    
                    year_val = bib.get('pub_year', '')
                    year_str = str(year_val) if year_val else ''
                    
                    abstract = bib.get('abstract', '')
                    
                    link_val = paper_obj.get('pub_url', paper_obj.get('eprint_url', ''))

                    paper_entry = {
                        'title': title.strip(),
                        'link': str(link_val).strip(),
                        'authors': authors_str.strip(),
                        'year': year_str.strip(),
                        'source': 'Google Scholar',
                        'original_abstract': abstract.strip(),
                    }
                    results.append(paper_entry)
                    logger.info(f"Scraped from Google Scholar: {title}")
                    count += 1
                except Exception as e_paper:
                    logger.error(f"Error processing a Google Scholar paper entry for query '{query}': {e_paper}", exc_info=True)
                time.sleep(API_CALL_DELAY)
            
        except Exception as e_search:
            logger.error(f"Error during Google Scholar search for query '{query}': {e_search}", exc_info=True)
            if "captcha" in str(e_search).lower() or "429" in str(e_search):
                logger.warning("Google Scholar CAPTCHA or rate limit hit. Consider manual search or longer delays.")

        logger.info(f"Total papers scraped from Google Scholar for query '{query}': {len(results)}")
        return results

def deduplicate_results(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen_identifiers = set()
    unique_results = []
    
    for result in results:
        title = str(result.get('title', '')).lower().strip()
        normalized_title = re.sub(r'[^a-z0-9\s]', '', title)
        normalized_title = re.sub(r'\s+', ' ', normalized_title).strip()
        
        link = str(result.get('link', '')).strip()
        
        identifier = link if link else normalized_title
        
        if not identifier:
            continue
            
        if identifier not in seen_identifiers:
            seen_identifiers.add(identifier)
            unique_results.append(result)
        else:
            logger.info(f"Removed duplicate: {result.get('title', 'N/A')} (Identifier: {identifier[:50]}...)")
            
    logger.info(f"Deduplication complete. Input: {len(results)}, Output: {len(unique_results)}")
    return unique_results 