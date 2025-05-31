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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PubMedSearchTool(BaseTool):
    name: str = "PubMedSearchTool"
    description: str = "A tool for searching and retrieving papers from PubMed"
    base_url: str = Field(default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")
    api_key: str = Field(default="")
    email: str = Field(default="")
    max_results: int = Field(default=5)

    def _run(self, query: str) -> List[Dict[str, str]]:
        logger.info(f"Starting PubMed search for query: {query}, max_results: {self.max_results}")
        
        # Configure Entrez
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key
        
        # First, search for paper IDs
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': self.max_results,
            'retmode': 'json',
            'api_key': self.api_key
        }
        
        search_url = f"{self.base_url}esearch.fcgi"
        response = requests.get(search_url, params=search_params)
        search_data = response.json()
        
        if 'esearchresult' not in search_data or 'idlist' not in search_data['esearchresult']:
            logger.error("No results found in PubMed search")
            return []
        
        paper_ids = search_data['esearchresult']['idlist']
        papers = []
        
        # Then fetch details for each paper
        for paper_id in paper_ids:
            fetch_params = {
                'db': 'pubmed',
                'id': paper_id,
                'retmode': 'xml',
                'api_key': self.api_key
            }
            
            fetch_url = f"{self.base_url}efetch.fcgi"
            response = requests.get(fetch_url, params=fetch_params)
            soup = BeautifulSoup(response.text, 'xml')
            
            article = soup.find('PubmedArticle')
            if article:
                # Extract title
                title = article.find('ArticleTitle')
                title_text = title.text if title else ''
                
                # Extract authors
                authors = article.find_all('Author')
                author_list = []
                for author in authors:
                    last_name = author.find('LastName')
                    fore_name = author.find('ForeName')
                    if last_name and fore_name:
                        author_list.append(f"{last_name.text} {fore_name.text}")
                authors_text = '; '.join(author_list)
                
                # Extract year
                pub_date = article.find('PubDate')
                year = ''
                if pub_date:
                    year_elem = pub_date.find('Year')
                    year = year_elem.text if year_elem else ''
                
                # Extract abstract
                abstract = article.find('Abstract')
                abstract_text = ''
                if abstract:
                    abstract_text = ' '.join([text.text for text in abstract.find_all('AbstractText')])
                
                paper = {
                    'title': title_text.strip(),
                    'authors': authors_text.strip(),
                    'year': year.strip(),
                    'source': 'pubmed',
                    'densified_abstract': abstract_text.strip(),
                    'keywords': '',
                    'relevance': ''
                }
                logger.info(f"Scraped paper: {paper['title']} (PubMed)")
                papers.append(paper)
            time.sleep(1)
        
        logger.info(f"Total papers scraped from PubMed: {len(papers)}")
        return papers

class GoogleScholarSearchTool(BaseTool):
    name: str = "Google Scholar Search Tool"
    description: str = "Search for papers on Google Scholar"
    max_results: int = Field(default=10)
    
    def _run(self, query: str) -> List[Dict[str, str]]:
        """Run the tool."""
        logger.info(f"Starting Google Scholar scrape for query: {query}, max_results: {self.max_results}")
        results = []
        
        # Set up proxy to avoid captcha
        try:
            pg = ProxyGenerator()
            pg.FreeProxies()
            scholarly.use_proxy(pg)
            logger.info("Using free proxies for Google Scholar searches")
        except Exception as e:
            logger.warning(f"Failed to set up proxy for Google Scholar: {e}")
        
        try:
            # Search for papers
            search_query = scholarly.search_pubs(query)
            
            for i in range(self.max_results):
                try:
                    paper = next(search_query)
                    if not paper:
                        break
                        
                    # Extract paper data safely
                    if hasattr(paper, 'bib'):
                        paper_data = paper.bib
                    else:
                        paper_data = paper.get('bib', {})
                    
                    # Create paper entry with all required fields
                    paper_entry = {
                        'title': paper_data.get('title', ''),
                        'authors': paper_data.get('author', 'Unknown'),
                        'year': paper_data.get('pub_year', 'Unknown'),
                        'source': 'Google Scholar',
                        'densified_abstract': paper_data.get('abstract', ''),
                        'keywords': [],
                        'relevance': ''
                    }
                    
                    # Only add papers with valid titles
                    if paper_entry['title']:
                        results.append(paper_entry)
                        
                except Exception as e:
                    err_msg = str(e).lower()
                    # Detect captcha or rate-limit
                    if 'captcha' in err_msg or 'too many requests' in err_msg or '429' in err_msg:
                        # Open browser for manual captcha solving
                        try:
                            import webbrowser, requests as _req
                            url = f"https://scholar.google.com/scholar?hl=en&q={_req.utils.quote(query)}"
                            logger.info(f"Opening browser to solve captcha: {url}")
                            webbrowser.open(url)
                            input("Please solve the captcha in your browser, then press Enter here to continue scraping...")
                            # Restart search
                            search_query = scholarly.search_pubs(query)
                            continue
                        except Exception as browser_e:
                            logger.error(f"Failed to open browser for captcha: {browser_e}")
                            break
                    logger.error(f"Error processing paper: {str(e)}")
                    continue
                time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error in Google Scholar search: {str(e)}")
            
        logger.info(f"Total papers scraped from Google Scholar: {len(results)}")
        return results

def deduplicate_results(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate papers based on title similarity."""
    seen_titles = set()
    unique_results = []
    
    for result in results:
        # Normalize title for comparison
        title = result['title'].lower().strip()
        title = ''.join(c for c in title if c.isalnum() or c.isspace())
        
        if title not in seen_titles:
            seen_titles.add(title)
            unique_results.append(result)
        else:
            logger.info(f"Removed duplicate: {result['title']}")
            
    return unique_results 