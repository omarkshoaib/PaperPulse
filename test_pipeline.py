import logging
from research_tools import ArxivSearchTool, GoogleScholarSearchTool, PubMedSearchTool, CSVWriterTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pipeline():
    # Initialize tools
    arxiv_tool = ArxivSearchTool(max_results=2)
    scholar_tool = GoogleScholarSearchTool(max_results=2)
    pubmed_tool = PubMedSearchTool(max_results=2)
    csv_writer = CSVWriterTool()
    
    # Test query
    query = "graph convolutional networks for action recognition"
    
    # Test arXiv search
    logger.info("Testing arXiv search...")
    try:
        arxiv_results = arxiv_tool._run(query)
        logger.info(f"Found {len(arxiv_results)} papers from arXiv")
        for paper in arxiv_results:
            csv_writer._run(paper)
    except Exception as e:
        logger.error(f"arXiv search failed: {str(e)}", exc_info=True)
    
    # Test Google Scholar search
    logger.info("Testing Google Scholar search...")
    try:
        scholar_results = scholar_tool._run(query)
        logger.info(f"Found {len(scholar_results)} papers from Google Scholar")
        for paper in scholar_results:
            csv_writer._run(paper)
    except Exception as e:
        logger.error(f"Google Scholar search failed: {str(e)}", exc_info=True)
    
    # Test PubMed search
    logger.info("Testing PubMed search...")
    try:
        pubmed_results = pubmed_tool._run(query)
        logger.info(f"Found {len(pubmed_results)} papers from PubMed")
        for paper in pubmed_results:
            csv_writer._run(paper)
    except Exception as e:
        logger.error(f"PubMed search failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_pipeline() 