import requests
from bs4 import BeautifulSoup
import csv
import os
import logging
from typing import List, Dict, Any, Union, Optional
from crewai.tools import BaseTool
from pydantic import Field, BaseModel, validator
from datetime import datetime
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperData(BaseModel):
    """Model for paper data validation"""
    title: str = ""
    link: str = ""
    authors: str = ""
    year: str = ""
    source: str = ""
    original_abstract: str = ""
    densified_abstract: str = ""
    keywords: str = ""
    relevance: str = ""
    FullTextPath: Optional[str] = ""

    @validator('*', pre=True, always=True)
    def ensure_str(cls, v):
        return str(v) if v is not None else ""

class ArxivSearchTool(BaseTool):
    name: str = "ArxivSearchTool"
    description: str = "A tool for searching and retrieving papers from arXiv"
    base_url: str = Field(default="http://export.arxiv.org/api/query")
    max_results: int = Field(default=5)

    def _run(self, query: str) -> List[Dict[str, str]]:
        logger.info(f"Starting arXiv scrape for query: {query}, max_results: {self.max_results}")
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': self.max_results
        }
        
        response = requests.get(self.base_url, params=params)
        soup = BeautifulSoup(response.text, 'xml')
        
        papers = []
        for entry in soup.find_all('entry'):
            authors_list = [author.find('name').text for author in entry.find_all('author')]
            paper = {
                'title': entry.title.text.strip(),
                'link': entry.link['href'],
                'original_abstract': entry.summary.text.strip(),
                'authors': ", ".join(authors_list),
                'year': entry.published.text.split('-')[0],
                'source': 'arXiv'
            }
            logger.info(f"Scraped paper: {paper['title']} ({paper['link']})")
            papers.append(paper)
            time.sleep(2)
        logger.info(f"Total papers scraped from arXiv: {len(papers)}")
        
        return papers

class CSVWriterTool(BaseTool):
    name: str = "CSVWriterTool"
    description: str = "A tool for writing research results to a CSV file"
    csv_filename: str = Field(default="research_results.csv")
    backup_dir: str = Field(default="backups")

    FIELDNAMES: List[str] = [
        'Title', 'Link', 'Authors', 'Year', 'Source', 
        'Original Abstract', 'Densified Abstract', 'Keywords', 'Relevance', 'Timestamp',
        'FullTextPath'
    ]

    def __init__(self, **data):
        super().__init__(**data)
        self._ensure_backup_dir()
        if not os.path.exists(self.csv_filename) or os.path.getsize(self.csv_filename) == 0:
            self._initialize_csv_with_headers()
        else:
            logger.info(f"CSV file '{self.csv_filename}' already exists and is not empty. Will append.")

    def _ensure_backup_dir(self):
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            logger.info(f"Created backup directory: {self.backup_dir}")

    def _backup_existing_file(self):
        if os.path.exists(self.csv_filename):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(self.backup_dir, f"{os.path.splitext(self.csv_filename)[0]}_{timestamp}.csv")
            try:
                os.rename(self.csv_filename, backup_file)
                logger.info(f"Created backup of existing file: {backup_file}")
                return True
            except Exception as e:
                logger.error(f"Failed to create backup for {self.csv_filename}: {e}")
        return False

    def _initialize_csv_with_headers(self):
        try:
            should_write_new = False
            if os.path.exists(self.csv_filename) and os.path.getsize(self.csv_filename) > 0:
                if self._backup_existing_file():
                    should_write_new = True
                else:
                    logger.warning(f"File {self.csv_filename} exists but was not backed up. Will not overwrite headers.")
                    return
            elif not os.path.exists(self.csv_filename):
                should_write_new = True
            
            if should_write_new:
                logger.info(f"Initializing new CSV file with headers: {self.csv_filename}")
                with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                    writer.writeheader()
                logger.info(f"Initialized CSV file '{self.csv_filename}' with headers.")

        except Exception as e:
            logger.error(f"Failed to initialize CSV file '{self.csv_filename}': {e}")

    def _validate_and_prepare_paper_for_csv(self, paper_data_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            mapped_input = {
                'title': paper_data_dict.get('title', paper_data_dict.get('Title', '')),
                'link': paper_data_dict.get('link', paper_data_dict.get('Link', '')),
                'authors': paper_data_dict.get('authors', paper_data_dict.get('Authors', '')),
                'year': paper_data_dict.get('year', paper_data_dict.get('Year', '')),
                'source': paper_data_dict.get('source', paper_data_dict.get('Source', '')),
                'original_abstract': paper_data_dict.get('original_abstract', paper_data_dict.get('Original Abstract', '')),
                'densified_abstract': paper_data_dict.get('densified_abstract', paper_data_dict.get('Densified Abstract', '')),
                'keywords': paper_data_dict.get('keywords', paper_data_dict.get('Keywords', '')),
                'relevance': paper_data_dict.get('relevance', paper_data_dict.get('Relevance', '')),
                'FullTextPath': paper_data_dict.get('FullTextPath', '')
            }
            validated_model = PaperData(**mapped_input)
            
            csv_row = {
                'Title': validated_model.title,
                'Link': validated_model.link,
                'Authors': validated_model.authors,
                'Year': validated_model.year,
                'Source': validated_model.source,
                'Original Abstract': validated_model.original_abstract,
                'Densified Abstract': validated_model.densified_abstract,
                'Keywords': validated_model.keywords,
                'Relevance': validated_model.relevance,
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'FullTextPath': validated_model.FullTextPath
            }
            return csv_row
        except Exception as e:
            title_for_error = paper_data_dict.get('title', paper_data_dict.get('Title', 'UNKNOWN'))
            logger.error(f"Data validation/preparation for CSV failed for title '{title_for_error}': {e}", exc_info=True)
            error_row = {field: "VALIDATION_ERROR" for field in self.FIELDNAMES}
            error_row['Title'] = title_for_error
            error_row['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return error_row

    def _run(self, paper_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        if isinstance(paper_data, dict):
            papers_to_write = [paper_data]
        elif isinstance(paper_data, list):
            papers_to_write = paper_data
        else:
            logger.error(f"Invalid paper_data type: {type(paper_data)}. Expected dict or list of dicts.")
            return False

        if not papers_to_write:
            logger.info("No paper data provided to write.")
            return True

        try:
            is_file_empty = not os.path.exists(self.csv_filename) or os.path.getsize(self.csv_filename) == 0

            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                
                if is_file_empty:
                     logger.info(f"File '{self.csv_filename}' is empty or new before appending; writing headers.")
                     writer.writeheader()
                
                for single_paper_dict in papers_to_write:
                    csv_row_dict = self._validate_and_prepare_paper_for_csv(single_paper_dict)
                    writer.writerow(csv_row_dict)
                    logger.info(f"Successfully wrote paper to CSV: {csv_row_dict.get('Title', 'UNKNOWN TITLE')}")
            return True
        except Exception as e:
            logger.error(f"Error writing to CSV '{self.csv_filename}': {e}", exc_info=True)
            return False

# Note: GoogleScholarSearchTool and PubMedSearchTool from search_tools.py are used by ResearchPipeline
# but their definitions are not in this file. This file focuses on Arxiv and CSV tools. 