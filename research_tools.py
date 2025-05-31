import requests
from bs4 import BeautifulSoup
import csv
import os
import logging
from typing import List, Dict, Any, Union
from crewai.tools import BaseTool
from pydantic import Field, BaseModel
from datetime import datetime
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperData(BaseModel):
    """Model for paper data validation"""
    link: str
    title: str
    densified_abstract: str
    keywords: str = ""  # Optional field
    relevance: str = ""  # Added relevance field

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
            paper = {
                'title': entry.title.text.strip(),
                'link': entry.link['href'],
                'abstract': entry.summary.text.strip()
            }
            logger.info(f"Scraped paper: {paper['title']} ({paper['link']})")
            papers.append(paper)
            time.sleep(2) # Added sleep
        logger.info(f"Total papers scraped: {len(papers)}")
        
        return papers

class CSVWriterTool(BaseTool):
    name: str = "CSVWriterTool"
    description: str = "A tool for writing research results to a CSV file"
    output_file: str = Field(default="research_results.csv")
    backup_dir: str = Field(default="backups")

    def __init__(self, **data):
        super().__init__(**data)
        self._ensure_backup_dir()
        self._initialize_csv()

    def _ensure_backup_dir(self):
        """Create backup directory if it doesn't exist"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def _backup_existing_file(self):
        """Create a backup of the existing CSV file if it exists"""
        if os.path.exists(self.output_file):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(self.backup_dir, f"research_results_{timestamp}.csv")
            try:
                os.rename(self.output_file, backup_file)
                logger.info(f"Created backup: {backup_file}")
            except Exception as e:
                logger.error(f"Failed to create backup: {str(e)}")

    def _initialize_csv(self):
        """Initialize the CSV file with headers"""
        try:
            # Create backup of existing file if it exists
            self._backup_existing_file()
            
            # Create new file with headers
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Paper Link', 'Title', 'Densified Abstract', 'Keywords', 'Relevance', 'Timestamp'])
            logger.info(f"Initialized CSV file: {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV file: {str(e)}")
            raise

    def _validate_paper_data(self, paper_data: Dict[str, str]) -> PaperData:
        """Validate and clean paper data"""
        try:
            # Ensure all required fields are present
            required_fields = ['link', 'title', 'densified_abstract', 'relevance']
            for field in required_fields:
                if field not in paper_data:
                    paper_data[field] = ''
            
            # Clean the data
            cleaned_data = {
                'link': str(paper_data.get('link', '')).strip(),
                'title': str(paper_data.get('title', '')).strip(),
                'densified_abstract': str(paper_data.get('densified_abstract', '')).strip(),
                'keywords': str(paper_data.get('keywords', '')).strip(),
                'relevance': str(paper_data.get('relevance', '')).strip()
            }
            
            return PaperData(**cleaned_data)
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise

    def _run(self, paper_data: Union[Dict[str, str], List[Dict[str, str]]]) -> bool:
        """
        Write paper data to CSV file
        Args:
            paper_data: Either a single paper dict or a list of paper dicts
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert single paper to list
            if isinstance(paper_data, dict):
                paper_data = [paper_data]
            
            # Validate and write each paper
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                for paper in paper_data:
                    # Validate and clean the data
                    validated_paper = self._validate_paper_data(paper)
                    
                    # Write to CSV
                    writer.writerow([
                        validated_paper.link,
                        validated_paper.title,
                        validated_paper.densified_abstract,
                        validated_paper.keywords,
                        validated_paper.relevance,
                        timestamp
                    ])
                    logger.info(f"Successfully wrote paper to CSV: {validated_paper.title}")
            
            return True
        except Exception as e:
            logger.error(f"Error writing to CSV: {str(e)}")
            return False 