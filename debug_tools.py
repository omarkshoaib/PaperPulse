import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("debug_tools.log", mode='w'),
        logging.StreamHandler()
    ]
)
# Now import the rest
import os
from research_tools import PubMedSearchTool, GoogleScholarSearchTool, ArxivSearchTool, CSVWriterTool

# Configure root logger
logger = logging.getLogger(__name__)

def test_search_tools():
    # Test query
    query = "human action recognition using pose estimation"
    
    # Initialize tools with explicit API keys
    pubmed = PubMedSearchTool(
        api_key="e8a81a517e5bf4934d54d0d854b1dcd0b408",
        user_email="omarkshoaib@gmail.com",
        max_results=2
    )
    
    scholar = GoogleScholarSearchTool(max_results=2)
    arxiv = ArxivSearchTool(max_results=2)
    csv_writer = CSVWriterTool()
    
    # Test PubMed
    logger.info("Testing PubMed Search Tool...")
    try:
        pubmed_results = pubmed._run(query)
        logger.info(f"PubMed returned {len(pubmed_results)} results")
        for paper in pubmed_results:
            logger.debug(f"PubMed paper: {paper.get('title')}")
            csv_writer._run(paper)
    except Exception as e:
        logger.error(f"PubMed search failed: {str(e)}", exc_info=True)
    
    # Test Google Scholar
    logger.info("Testing Google Scholar Search Tool...")
    try:
        scholar_results = scholar._run(query)
        logger.info(f"Google Scholar returned {len(scholar_results)} results")
        for paper in scholar_results:
            logger.debug(f"Google Scholar paper: {paper.get('title')}")
            csv_writer._run(paper)
    except Exception as e:
        logger.error(f"Google Scholar search failed: {str(e)}", exc_info=True)
    
    # Test arXiv
    logger.info("Testing arXiv Search Tool...")
    try:
        arxiv_results = arxiv._run(query)
        logger.info(f"arXiv returned {len(arxiv_results)} results")
        for paper in arxiv_results:
            logger.debug(f"arXiv paper: {paper.get('title')}")
            csv_writer._run(paper)
    except Exception as e:
        logger.error(f"arXiv search failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_search_tools() 