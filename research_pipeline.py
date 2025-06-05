from crewai import Agent, Task, Crew, Process
from research_tools import ArxivSearchTool, CSVWriterTool # Assuming PaperData is in research_tools
from typing import List, Dict, Any, Optional
from crewai import LLM
import os
import requests
import json
import time
import logging
import pandas as pd # For CSV processing
import re # Add re for robust JSON extraction
from Bio import Entrez # For PubMed full text download
from bs4 import BeautifulSoup # Added for parsing HTML
import csv
from urllib.parse import urljoin, urlparse # For constructing absolute URLs
from litellm.exceptions import RateLimitError # For handling LLM rate limits
from dotenv import load_dotenv # Add this import

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Default placeholder for LLM-generated fields
PENDING_LLM_PLACEHOLDER = "PENDING_LLM_PROCESSING"
DEFAULT_CSV_FILE = "GCN_DAM_2.csv"
DOWNLOAD_DIR = "downloaded_papers"

# API Rate Limiting Configuration
API_CALL_DELAY = int(os.getenv("API_CALL_DELAY", "90"))  # Seconds between API calls
DOWNLOAD_WAIT_TIME = int(os.getenv("DOWNLOAD_WAIT_TIME", "90"))  # Seconds to wait for Selenium downloads

# LLM Rate Limiting Configuration
LLM_CALL_DELAY = int(os.getenv("LLM_CALL_DELAY", "5"))  # Seconds between LLM calls (default: 5 seconds)
LLM_RETRY_DELAY = int(os.getenv("LLM_RETRY_DELAY", "30"))  # Seconds to wait before retrying after rate limit
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))  # Maximum number of retries for LLM calls

DOWNLOAD_STATUS_NO_PMCID = "NO_PMC_ID_FOUND"
DOWNLOAD_STATUS_NO_PDF_LINK = "NO_PDF_LINK_ON_PMC"
DOWNLOAD_STATUS_FAILED = "DOWNLOAD_FAILED"
DOWNLOAD_STATUS_NOT_PUBMED = "NOT_A_PUBMED_ENTRY"
DOWNLOAD_STATUS_NO_LINK = "NO_LINK_IN_CSV"
DOWNLOAD_STATUS_BAD_LINK = "MALFORMED_PUBMED_LINK"
DOWNLOAD_STATUS_HTML_CONTENT = "HTML_CONTENT_RETURNED"
DOWNLOAD_STATUS_PMC_FORBIDDEN = "PMC_FORBIDDEN"
DOWNLOAD_STATUS_PDF_FORBIDDEN = "PDF_FORBIDDEN"
DOWNLOAD_STATUS_PENDING = "PENDING_DOWNLOAD"
DOWNLOAD_STATUS_SUCCESS = "DOWNLOAD_SUCCESS"

class ResearchPipeline:
    def __init__(self, 
                 provider: Optional[str] = None, # LLM provider, optional at init
                 model_name: Optional[str] = None, # LLM model, optional at init
                 base_url: Optional[str] = None, # For Ollama, optional at init
                 pubmed_email: Optional[str] = None, # For PubMed tool
                 pubmed_api_key: Optional[str] = None): # For PubMed tool
        
        # LLM related attributes, initialized lazily
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        self.local_llm: Optional[LLM] = None
        self.summarizer_agent: Optional[Agent] = None
        self.validator_agent: Optional[Agent] = None
        self.query_generation_agent: Optional[Agent] = None # New agent
        self.llm_initialized = False

        # Tools for scraping (always initialized)
        self.pubmed_email = pubmed_email # Set from constructor argument
        self.pubmed_api_key = pubmed_api_key # Set from constructor argument
        self.arxiv_tool = ArxivSearchTool() 
        # Ensure CSVWriterTool in research_tools.py can accept csv_filename or has a default
        self.csv_tool = CSVWriterTool(csv_filename=DEFAULT_CSV_FILE) 

        try:
            from search_tools import PubMedSearchTool, GoogleScholarSearchTool, deduplicate_results
            # PubMedSearchTool will now use the email and api_key passed to ResearchPipeline,
            # which are sourced from .env/environment in the main block.
            self.pubmed_tool = PubMedSearchTool(
                email=self.pubmed_email, # Uses the value from the constructor
                api_key=self.pubmed_api_key # Uses the value from the constructor
            )
            self.google_scholar_tool = GoogleScholarSearchTool()
            self.deduplicate_results_func = deduplicate_results
            logger.info("PubMed and Google Scholar tools initialized for scraping.")
        except ImportError:
            logger.error("Could not import from search_tools.py. PubMed and Google Scholar searches will be disabled.")
            self.pubmed_tool = None
            self.google_scholar_tool = None
            # Provide a simple list passthrough if deduplication isn't available
            # This basic deduplication assumes 'link' is a unique identifier for a paper.
            self.deduplicate_results_func = lambda papers_list: list({p['link']: p for p in papers_list if p.get('link')}.values()) if papers_list else []
        
        # Create download directory if it doesn't exist
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)
            logger.info(f"Created directory for downloaded papers: {DOWNLOAD_DIR}")

    def _verify_ollama_setup(self, base_url: str, model_name: str):
        logger.info(f"Verifying Ollama setup: URL={base_url}, Model={model_name}")
        try:
            response = requests.get(f"{base_url}/api/tags") # Verify this is the correct Ollama API endpoint
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            models_data = response.json()
            available_models = [m.get('name') for m in models_data.get('models', []) if m.get('name')]
            if model_name not in available_models:
                logger.error(f"Model '{model_name}' not found in Ollama. Available: {available_models}")
                # Consider providing instructions to pull the model
                raise ValueError(f"Model '{model_name}' not found. Please pull it with 'ollama pull {model_name}'.")
            logger.info(f"Ollama setup verified. Model '{model_name}' is available.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Could not connect to Ollama server at {base_url}: {e}")
            raise ConnectionError(f"Ollama connection failed. Ensure Ollama is running (e.g., 'ollama serve'). Error: {e}")
        except ValueError as e: # Re-raise ValueError if model not found
            raise e
        except json.JSONDecodeError as e: # Handle cases where response is not valid JSON
            logger.error(f"Failed to decode JSON response from Ollama server at {base_url}/api/tags: {e}. Response text: {response.text}")
            raise ConnectionError(f"Invalid JSON response from Ollama server. Error: {e}")

    def _initialize_llm_resources(self):
        if self.llm_initialized:
            logger.info("LLM resources already initialized.")
            return

        if not self.provider or not self.model_name:
            logger.error("LLM provider and model name must be specified to initialize LLM resources.")
            # It might be better to raise an error if called without these being set
            # Or ensure they are set before this method is called by the control flow.
            raise ValueError("LLM provider and model name required for this operation.")

        logger.info(f"Initializing LLM resources with provider: {self.provider}, model: {self.model_name}")
        llm_config = {"temperature": 0.7, "timeout": 300}

        if self.provider == "ollama":
            if not self.base_url:
                self.base_url = "http://localhost:11434" # Default Ollama URL
                logger.info(f"Ollama provider selected, no base_url provided, defaulting to {self.base_url}")
            self._verify_ollama_setup(self.base_url, self.model_name)
            self.local_llm = LLM(model=f"ollama/{self.model_name}", base_url=self.base_url, **llm_config)
        elif self.provider == "gemini":
            gemini_model_str = self.model_name if self.model_name.startswith("gemini/") else f"gemini/{self.model_name}"
            self.local_llm = LLM(model=gemini_model_str, **llm_config)
            logger.info(f"Using Gemini model: {gemini_model_str}")
        elif self.provider == "huggingface":
            # Ensure model_name for HuggingFace is correctly formatted (e.g. "huggingface/RepoId/ModelId")
            # Or just the RepoId/ModelId if LiteLLM handles the prefix.
            hf_model_str = self.model_name if self.model_name.startswith("huggingface/") else f"huggingface/{self.model_name}"
            self.local_llm = LLM(model=hf_model_str, **llm_config)
            logger.info(f"Using HuggingFace model: {hf_model_str}")
        else:
            # This should ideally not be reached if provider/model_name are checked before calling
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Please choose from 'ollama', 'gemini', 'huggingface'.")

        # Initialize LLM-dependent agents
        self.summarizer_agent = Agent(
            role="Research Summarizer",
            goal="For each provided paper (title, link, original_abstract, etc.), create a concise yet dense summary of its original_abstract using the Chain of Density technique, and extract 3-5 relevant keywords.",
            backstory="You are an AI expert in scientific literature analysis. You can quickly grasp the core contributions of a paper from its abstract and identify its most salient keywords.",
            verbose=True, allow_delegation=False, llm=self.local_llm
        )
        self.validator_agent = Agent(
            role='Research Relevance Validator',
            goal='Critically evaluate if a research paper is relevant to a given project description based on its title, densified summary, and keywords.',
            backstory=("You are an AI assistant with a keen eye for detail and a strong understanding of research methodology. "
                       "Your task is to determine relevance by strictly outputting 'RELEVANT' or 'NOT RELEVANT'."),
            verbose=True, allow_delegation=False, llm=self.local_llm
        )
        self.query_generation_agent = Agent( # New Agent
            role="Scientific Query Generator",
            goal="Generate a list of 3-5 diverse and effective search query strings for academic paper databases (like arXiv, PubMed, Google Scholar) based on a given research project description. The queries should be targeted to find relevant literature.",
            backstory="You are an AI assistant skilled in understanding research topics and formulating precise search queries to retrieve relevant scientific papers. Your response MUST be a single valid JSON array of strings (e.g., [\"query 1\", \"query 2\", \"query 3\"]). The queries should be from different points of view and not just the same keywords and the queries shouldn't have any stop words.",
            verbose=True, allow_delegation=False, llm=self.local_llm
        )
        self.llm_initialized = True
        logger.info("LLM resources initialized successfully.")

    def _safe_llm_call_with_retry(self, crew: Crew, operation_name: str) -> Any:
        """
        Execute LLM crew with rate limiting and retry logic.
        
        Args:
            crew: The CrewAI crew to execute
            operation_name: Human-readable name for logging
            
        Returns:
            The crew result or None if all retries failed
        """
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    retry_delay = LLM_RETRY_DELAY * (2 ** (attempt - 1))  # Exponential backoff
                    logger.info(f"Retrying {operation_name} (attempt {attempt + 1}/{LLM_MAX_RETRIES + 1}) after {retry_delay}s delay...")
                    time.sleep(retry_delay)
                else:
                    # Always add a small delay between LLM calls to be respectful
                    time.sleep(LLM_CALL_DELAY)
                
                result = crew.kickoff()
                logger.debug(f"{operation_name} completed successfully on attempt {attempt + 1}")
                return result
                
            except RateLimitError as e:
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    if attempt < LLM_MAX_RETRIES:
                        # Extract retry delay from error if available
                        error_str = str(e)
                        retry_match = re.search(r'"retryDelay":\s*"(\d+)s"', error_str)
                        if retry_match:
                            suggested_delay = int(retry_match.group(1))
                            retry_delay = max(suggested_delay, LLM_RETRY_DELAY)
                        else:
                            retry_delay = LLM_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                        
                        logger.warning(f"{operation_name} hit rate limit on attempt {attempt + 1}. Waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"{operation_name} failed after {LLM_MAX_RETRIES + 1} attempts due to rate limiting: {e}")
                        return None
                else:
                    logger.error(f"{operation_name} failed with non-recoverable rate limit error: {e}")
                    return None
                    
            except Exception as e:
                if attempt < LLM_MAX_RETRIES:
                    retry_delay = LLM_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"{operation_name} failed on attempt {attempt + 1} with error: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"{operation_name} failed after {LLM_MAX_RETRIES + 1} attempts: {e}", exc_info=True)
                    return None
        
        return None

    def generate_search_queries(self, project_description: str) -> List[str]:
        self._initialize_llm_resources() 
        if not self.llm_initialized or not self.query_generation_agent:
            logger.error("LLM resources (including query generation agent) not initialized. Cannot generate queries.")
            # Raising an error is better than returning empty list to signal failure clearly
            raise RuntimeError("LLM resources not ready for query generation. Ensure provider/model are set correctly.")

        logger.info(f"Generating search queries for project: {project_description[:100]}...")

        task_description = (
            f"Based on the following research project description, generate a list of 3-5 diverse and "
            f"effective search query strings. These queries will be used to find relevant academic papers "
            f"on platforms like arXiv, PubMed, and Google Scholar. Your response MUST be a single valid JSON "
            f"array of strings (e.g., [\"query 1\", \"query 2\", \"query 3\"]). Do not include any other text "
            f"before or after the JSON array.\n"
            f"Project Description: \"{project_description}\""
        )
        query_gen_task = Task(
            description=task_description,
            agent=self.query_generation_agent,
            expected_output="A single JSON array of strings, where each string is a search query."
        )

        query_crew = Crew(agents=[self.query_generation_agent], tasks=[query_gen_task], verbose=False)
        crew_result_obj = self._safe_llm_call_with_retry(query_crew, "Query Generation")

        generated_queries: List[str] = []
        if crew_result_obj and hasattr(crew_result_obj, 'raw') and crew_result_obj.raw:
            try:
                raw_output = crew_result_obj.raw
                logger.debug(f"Raw output from query generator LLM: {raw_output}")
                # Attempt to find JSON array within the raw output
                match = re.search(r'\[\s*(?:\"(?:\\\"|[^\"])*\"\s*,\s*)*\"(?:\\\"|[^\"])*\"\s*\]', raw_output, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    parsed_list = json.loads(json_str)
                    if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                        generated_queries = [q.strip() for q in parsed_list if q.strip()] # Clean and filter empty
                        if generated_queries:
                             logger.info(f"Successfully generated and parsed queries: {generated_queries}")
                        else:
                             logger.warning(f"Query generation resulted in an empty list of queries after cleaning. Raw: {raw_output[:200]}...")
                    else:
                        logger.error(f"Parsed JSON from query generator is not a list of strings: {parsed_list}. Raw: {raw_output[:200]}...")
                else:
                    logger.error(f"No valid JSON array found in query generator output. Raw: {raw_output[:200]}...")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from query generator: {e}. Raw: {crew_result_obj.raw[:200]}...")
            except Exception as e: # Catch any other unexpected errors during parsing
                logger.error(f"Unexpected error processing query generator output: {e}. Raw: {crew_result_obj.raw[:200]}...", exc_info=True)
        else:
            logger.warning("Query generation task produced no raw output or an empty result.")
        
        if not generated_queries:
            logger.warning("Query generation failed or produced no valid queries. Returning empty list.")
            
        return generated_queries

    def search_papers(self, search_queries: List[str], max_results_per_source: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Starting paper search for {len(search_queries)} queries, max {max_results_per_source} results per source.")
        all_found_papers: List[Dict[str, Any]] = [] # Ensure type hint for clarity

        # Update max_results for each tool instance if they have such an attribute
        # This assumes tools have a 'max_results' attribute that can be set.
        # If not, this part might need adjustment based on how each tool handles result limits.
        if hasattr(self.arxiv_tool, 'max_results'): self.arxiv_tool.max_results = max_results_per_source
        if self.pubmed_tool and hasattr(self.pubmed_tool, 'max_results'): self.pubmed_tool.max_results = max_results_per_source
        if self.google_scholar_tool and hasattr(self.google_scholar_tool, 'max_results'): self.google_scholar_tool.max_results = max_results_per_source

        for query in search_queries:
            logger.info(f"Searching for query: '{query}'")
            try:
                logger.info(f"Searching arXiv for: {query}")
                arxiv_results = self.arxiv_tool._run(query=query)
                logger.info(f"Found {len(arxiv_results)} papers from arXiv for query '{query}'.")
                all_found_papers.extend(arxiv_results)
            except Exception as e:
                logger.error(f"Error during arXiv search for query '{query}': {e}", exc_info=True)

            if self.pubmed_tool:
                try:
                    logger.info(f"Searching PubMed for: {query}")
                    pubmed_results = self.pubmed_tool._run(query=query)
                    logger.info(f"Found {len(pubmed_results)} papers from PubMed for query '{query}'.")
                    all_found_papers.extend(pubmed_results)
                except Exception as e:
                    logger.error(f"Error during PubMed search for query '{query}': {e}", exc_info=True)
            
            if self.google_scholar_tool:
                try:
                    logger.info(f"Searching Google Scholar for: {query}")
                    gs_results = self.google_scholar_tool._run(query=query)
                    logger.info(f"Found {len(gs_results)} papers from Google Scholar for query '{query}'.")
                    all_found_papers.extend(gs_results)
                except Exception as e:
                    logger.error(f"Error during Google Scholar search for query '{query}': {e}", exc_info=True)
        
        logger.info(f"Total papers found across all sources before deduplication: {len(all_found_papers)}")
        
        unique_papers = self.deduplicate_results_func(all_found_papers)
        logger.info(f"Total unique papers after deduplication: {len(unique_papers)}")
        return unique_papers

    def scrape_and_save_raw_papers(self, search_queries: List[str], max_results_per_source: int = 5):
        logger.info("Starting raw paper scraping process...")
        if not search_queries:
            logger.warning("No search queries provided. Nothing to scrape.")
            return

        papers_to_save = self.search_papers(search_queries, max_results_per_source)
        
        if not papers_to_save:
            logger.info("No papers found from any source for the given queries.")
            return

        count = 0
        for paper in papers_to_save:
            # Normalize author data to string if it's a list (common from some tools)
            authors_data = paper.get('authors', [])
            if isinstance(authors_data, list):
                authors_str = ', '.join(authors_data)
            elif isinstance(authors_data, str):
                authors_str = authors_data
            else:
                authors_str = '' # Default to empty string if type is unexpected
            
            # Ensure all required keys are present for CSV, using PENDING_LLM for LLM fields
            paper_data_for_csv = {
                'title': paper.get('title', 'N/A'),
                'link': paper.get('link', ''),
                # Use 'original_abstract' if present, otherwise empty string
                'original_abstract': paper.get('original_abstract', ''), 
                'authors': authors_str,
                'year': str(paper.get('year', '')), # Ensure year is string
                'source': paper.get('source', 'Unknown'),
                'densified_abstract': PENDING_LLM_PLACEHOLDER,
                'keywords': PENDING_LLM_PLACEHOLDER,
                'relevance': PENDING_LLM_PLACEHOLDER
            }
            try:
                self.csv_tool._run(paper_data=paper_data_for_csv)
                count += 1
                logger.info(f"Saved raw data for paper: {paper_data_for_csv['title']}")
            except Exception as e:
                logger.error(f"Error saving raw paper '{paper_data_for_csv['title']}' to CSV: {e}", exc_info=True)
        
        logger.info(f"Raw scraping complete. Saved {count} papers to {self.csv_tool.csv_filename if hasattr(self.csv_tool, 'csv_filename') else DEFAULT_CSV_FILE}.")

    def process_papers_from_csv(self, project_description: str, csv_filepath: str = DEFAULT_CSV_FILE):
        self._initialize_llm_resources() # Ensure LLMs are ready
        if not self.llm_initialized or not self.summarizer_agent or not self.validator_agent:
             logger.error("LLM resources not initialized. Cannot process papers. Please ensure provider and model_name are set.")
             return

        logger.info(f"Starting LLM processing for papers in '{csv_filepath}' relevant to: '{project_description[:100]}...'")
        try:
            # Specify dtype to prevent type inference issues, esp. for potentially empty LLM fields
            df = pd.read_csv(csv_filepath, dtype=str).fillna(value='') # Read all as str, fill NaN with empty string
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_filepath}. Please scrape papers first (Option 1).")
            return
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {csv_filepath}. Nothing to process.")
            return
        
        # Ensure all expected columns exist, creating them if necessary
        expected_cols = [ # These must match CSVWriterTool.FIELDNAMES exactly
            'Title', 'Link', 'Authors', 'Year', 'Source', 
            'Original Abstract', 'Densified Abstract', 'Keywords', 'Relevance'
            # 'Timestamp' is in FIELDNAMES but not strictly needed for this processing logic directly
        ]
        for col in expected_cols:
            if col not in df.columns:
                logger.info(f"Column '{col}' not found in CSV, adding it.")
                df[col] = PENDING_LLM_PLACEHOLDER if col in ['Densified Abstract', 'Keywords', 'Relevance'] else ''
        
        # Ensure original_abstract is not NaN and is a string for LLM processing
        df['Original Abstract'] = df['Original Abstract'].astype(str).fillna('')
        # Initialize LLM fields if they are completely empty or NaN, to ensure they are strings for comparison
        for col in ['Densified Abstract', 'Keywords', 'Relevance']:
            df[col] = df[col].astype(str).fillna(PENDING_LLM_PLACEHOLDER)
            df[col] = df[col].apply(lambda x: PENDING_LLM_PLACEHOLDER if not x.strip() else x)

        papers_processed_count = 0
        papers_to_update_indices = []

        for index, row in df.iterrows():
            # Check if densified_abstract indicates a need for processing
            current_densified = str(row.get('Densified Abstract', '')).strip()
            if current_densified == PENDING_LLM_PLACEHOLDER or not current_densified:
                papers_to_update_indices.append(index)
        
        if not papers_to_update_indices:
            logger.info("No papers found in the CSV that require LLM processing.")
            return

        logger.info(f"Found {len(papers_to_update_indices)} papers to process with LLM.")

        for index in papers_to_update_indices:
            row = df.loc[index]
            title = str(row.get('Title', 'N/A'))
            original_abstract_text = str(row.get('Original Abstract', '')).strip()

            if not original_abstract_text:
                logger.warning(f"Skipping paper '{title}' due to empty original_abstract.")
                df.loc[index, 'Densified Abstract'] = "SKIPPED_EMPTY_ABSTRACT"
                df.loc[index, 'Keywords'] = "SKIPPED_EMPTY_ABSTRACT"
                df.loc[index, 'Relevance'] = "SKIPPED_EMPTY_ABSTRACT"
                continue

            logger.info(f"Processing paper ({papers_processed_count + 1}/{len(papers_to_update_indices)}): {title}")
            
            # Prepare paper data for the summarizer agent
            paper_for_llm = {
                "title": title,
                "link": str(row.get('Link', '')),
                "original_abstract": original_abstract_text,
                "source": str(row.get('Source', '')),
                "authors": str(row.get('Authors', '')),
                "year": str(row.get('Year', ''))
            }

            # Summarization Task
            summarize_task_desc = (
                f"For the following paper, create a dense summary of its original_abstract "
                f"and extract 3-5 relevant keywords. Your response MUST be a single valid JSON object with two keys: "
                f"\"densified_abstract\" (string) and \"keywords\" (a list of strings).\n"
                f"Paper details (use this for context): {json.dumps(paper_for_llm)}"
            )
            summarize_task = Task(
                description=summarize_task_desc,
            agent=self.summarizer_agent,
                expected_output="A single JSON object with keys 'densified_abstract' and 'keywords'."
            )
            
            summary_crew = Crew(agents=[self.summarizer_agent], tasks=[summarize_task], verbose=False)
            summary_result_obj = self._safe_llm_call_with_retry(summary_crew, "Summarization")

            densified_abstract_val = "ERROR_SUMMARIZING"
            keywords_str_val = "ERROR_SUMMARIZING"

            if summary_result_obj is None:
                logger.error(f"Summarization failed for '{title}' after all retry attempts.")
                densified_abstract_val = "FAILED_AFTER_RETRIES"
                keywords_str_val = "FAILED_AFTER_RETRIES"
            elif summary_result_obj and hasattr(summary_result_obj, 'raw') and summary_result_obj.raw:
                try:
                    # Attempt to extract JSON from potentially messy LLM output
                    raw_output = summary_result_obj.raw
                    json_match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        summary_data = json.loads(json_str)
                        densified_abstract_val = summary_data.get('densified_abstract', "ERROR_PARSING_SUMMARY")
                        keywords_list = summary_data.get('keywords', [])
                        keywords_str_val = ', '.join(keywords_list) if isinstance(keywords_list, list) and keywords_list else "NO_KEYWORDS_EXTRACTED"
                    else:
                        logger.error(f"No JSON object found in summarizer output for '{title}'. Raw: {raw_output[:200]}...")
                        densified_abstract_val = "ERROR_NO_JSON_IN_SUMMARY_OUTPUT"
                        keywords_str_val = "ERROR_NO_JSON_IN_SUMMARY_OUTPUT"

                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding summarizer JSON for '{title}': {e}. Raw: {summary_result_obj.raw[:200]}...")
                except Exception as e:
                     logger.error(f"Unexpected error processing summary for '{title}': {e}", exc_info=True)
            else:
                logger.warning(f"Summarization task produced no raw output for '{title}'.")

            df.loc[index, 'Densified Abstract'] = densified_abstract_val
            df.loc[index, 'Keywords'] = keywords_str_val
            logger.info(f"Summarized '{title}'. Keywords: {keywords_str_val[:100]}...")
            
            # Validation Task
            relevance_val = "ERROR_VALIDATING"
            if densified_abstract_val not in ["ERROR_SUMMARIZING", "ERROR_PARSING_SUMMARY", "ERROR_NO_JSON_IN_SUMMARY_OUTPUT", "SKIPPED_EMPTY_ABSTRACT", "FAILED_AFTER_RETRIES"]:
                validation_task_desc = (
                    f"Critically evaluate if the research paper is relevant to the project description: '{project_description}'.\n"
                    f"Paper Details:\nTitle: {title}\nSummary: {densified_abstract_val}\nKeywords: {keywords_str_val}\n"
                    f"Your answer MUST be exactly 'RELEVANT' or 'NOT RELEVANT'. Do not provide any other explanation, text, or punctuation."
                )
                validation_task = Task(
                    description=validation_task_desc,
                    agent=self.validator_agent,
                    expected_output="A single word: 'RELEVANT' or 'NOT RELEVANT'."
                )
                validation_crew = Crew(agents=[self.validator_agent], tasks=[validation_task], verbose=False)
                validation_result_obj = self._safe_llm_call_with_retry(validation_crew, "Validation")
                
                if validation_result_obj is None:
                    logger.error(f"Validation failed for '{title}' after all retry attempts.")
                    relevance_val = "FAILED_AFTER_RETRIES"
                elif validation_result_obj and hasattr(validation_result_obj, 'raw') and validation_result_obj.raw:
                    relevance_val = str(validation_result_obj.raw).strip().upper()
                    if relevance_val not in ["RELEVANT", "NOT RELEVANT"]:
                        logger.warning(f"Unexpected validation output for '{title}': '{relevance_val}'. Marking as PENDING_REVIEW.")
                        relevance_val = "PENDING_REVIEW"
                else:
                    logger.warning(f"Validation task produced no raw output for '{title}'.")
            else:
                logger.info(f"Skipping validation for '{title}' due to summarization error or empty abstract.")
            
            df.loc[index, 'Relevance'] = relevance_val
            logger.info(f"Validated '{title}' as: {relevance_val}")
            papers_processed_count += 1
            
            # Save after each processed paper to prevent data loss on long runs / errors
            try:
                df.to_csv(csv_filepath, index=False)
                if papers_processed_count % 1 == 0: # Log every paper, or change to 5 or 10 for less verbosity
                    logger.info(f"Intermediate save to '{csv_filepath}' after processing {papers_processed_count} paper(s).")
            except Exception as e:
                logger.error(f"Error during intermediate save to CSV: {e}", exc_info=True)

        logger.info(f"LLM processing complete. Attempted to update {papers_processed_count} paper(s) in '{csv_filepath}'.")
        # Final save is done inside the loop, this is just a confirmation log.

    def scrape_papers_from_identifiers_csv(self, input_csv_path: str, pmid_col: str = 'pmid', doi_col: str = 'doi'):
        logger.info(f"Starting paper scraping from input CSV: {input_csv_path}")
        try:
            # Try to infer encoding, common ones first
            try:
                input_df = pd.read_csv(input_csv_path, dtype=str).fillna('')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed for {input_csv_path}, trying latin1.")
                input_df = pd.read_csv(input_csv_path, dtype=str, encoding='latin1').fillna('')
        except FileNotFoundError:
            logger.error(f"Input CSV file not found: {input_csv_path}")
            print(f"Error: Input CSV file '{input_csv_path}' not found.")
            return
        except Exception as e:
            logger.error(f"Error reading input CSV '{input_csv_path}': {e}", exc_info=True)
            print(f"Error: Could not read input CSV '{input_csv_path}'. Check logs.")
            return

        # Normalize column names for lookup (case-insensitive)
        original_columns = list(input_df.columns)
        input_df.columns = [str(col).lower() for col in input_df.columns]
        pmid_col_lower = pmid_col.lower()
        doi_col_lower = doi_col.lower()

        actual_pmid_col = None
        actual_doi_col = None

        # Find the actual column names (case-insensitive match)
        for i, col_lower in enumerate(input_df.columns):
            if col_lower == pmid_col_lower:
                actual_pmid_col = original_columns[i]
            if col_lower == doi_col_lower:
                actual_doi_col = original_columns[i]
        
        # Re-read with original column names if found, to use for .get()
        if actual_pmid_col or actual_doi_col:
             try:
                input_df = pd.read_csv(input_csv_path, dtype=str).fillna('') # Read again to use original case for .get()
             except UnicodeDecodeError:
                input_df = pd.read_csv(input_csv_path, dtype=str, encoding='latin1').fillna('')
             except Exception:
                 pass # If re-read fails, continue with lowercased column access as fallback (less ideal)

        if not actual_pmid_col and not actual_doi_col:
            logger.error(f"Neither PMID column (tried '{pmid_col}') nor DOI column (tried '{doi_col}') found in input CSV. Available columns (lowercase): {list(input_df.columns)}")
            print(f"Error: Input CSV must contain a column for PMID (e.g., '{pmid_col}') or DOI (e.g., '{doi_col}').")
            return

        all_papers_from_csv: List[Dict[str, Any]] = []
        processed_rows = 0
        found_papers_count = 0

        for index, row in input_df.iterrows():
            processed_rows += 1
            paper_data_for_row = None
            pmid_value = row.get(actual_pmid_col, '').strip() if actual_pmid_col else ''
            doi_value = row.get(actual_doi_col, '').strip() if actual_doi_col else ''

            logger.info(f"Processing row {index + 1}: PMID='{pmid_value if pmid_value else 'N/A'}', DOI='{doi_value if doi_value else 'N/A'}'")

            if not pmid_value and not doi_value:
                logger.warning(f"Skipping row {index + 1} as both PMID and DOI are missing or empty.")
                continue

            # 1. Try PMID with PubMedTool (if PMID exists)
            if pmid_value and self.pubmed_tool:
                try:
                    logger.info(f"Searching PubMed with PMID: {pmid_value}")
                    pubmed_results = self.pubmed_tool._run(query=pmid_value)
                    if pubmed_results:
                        paper_data_for_row = pubmed_results[0] 
                        # Source is usually set by the tool, ensure it's there
                        paper_data_for_row['source'] = paper_data_for_row.get('source', 'PubMed') 
                        logger.info(f"Found paper via PubMed (PMID: {pmid_value}): {paper_data_for_row.get('title')}") # Changed 'Title' to 'title'
                except Exception as e:
                    logger.error(f"Error searching PubMed with PMID '{pmid_value}': {e}", exc_info=True)
            
            if not paper_data_for_row and doi_value:
                logger.info(f"PMID search failed or no PMID. Trying DOI: {doi_value}")
                if self.google_scholar_tool:
                    try:
                        logger.info(f"Searching Google Scholar with DOI: {doi_value}")
                        gs_results = self.google_scholar_tool._run(query=doi_value)
                        if gs_results:
                            paper_data_for_row = gs_results[0]
                            paper_data_for_row['source'] = paper_data_for_row.get('source', 'Google Scholar via DOI')
                            logger.info(f"Found paper via Google Scholar (DOI: {doi_value}): {paper_data_for_row.get('title')}") # Changed to .get('title')
                    except Exception as e:
                        logger.error(f"Error searching Google Scholar with DOI '{doi_value}': {e}", exc_info=True)
                
                if not paper_data_for_row and self.arxiv_tool:
                    try:
                        logger.info(f"Searching arXiv with DOI: {doi_value}")
                        arxiv_results = self.arxiv_tool._run(query=doi_value)
                        if arxiv_results:
                            paper_data_for_row = arxiv_results[0]
                            paper_data_for_row['source'] = paper_data_for_row.get('source', 'arXiv via DOI')
                            logger.info(f"Found paper via arXiv (DOI: {doi_value}): {paper_data_for_row.get('title')}") # Changed to .get('title')
                    except Exception as e:
                        logger.error(f"Error searching arXiv with DOI '{doi_value}': {e}", exc_info=True)
            
            if paper_data_for_row:
                # Prepare data with all necessary fields using lowercase keys for deduplication and CSV writing
                prep_data_for_dedup = {
                    'title': paper_data_for_row.get('title', 'N/A'),
                    'link': paper_data_for_row.get('link', ''),
                    'authors': paper_data_for_row.get('authors', ''),
                    'year': str(paper_data_for_row.get('year', '')),
                    'source': paper_data_for_row.get('source', 'Identifier CSV'), # Source is set above
                    'original_abstract': paper_data_for_row.get('original_abstract', ''),
                    'densified_abstract': PENDING_LLM_PLACEHOLDER,
                    'keywords': PENDING_LLM_PLACEHOLDER,
                    'relevance': PENDING_LLM_PLACEHOLDER,
                    'FullTextPath': '' # Initialize for potential future download
                }
                all_papers_from_csv.append(prep_data_for_dedup)
                found_papers_count += 1
            else:
                logger.warning(f"Could not find paper for row {index + 1} (PMID: {pmid_value}, DOI: {doi_value})")

        logger.info(f"Processed {processed_rows} rows from input CSV. Found details for {found_papers_count} papers.")

        if all_papers_from_csv:
            logger.info(f"Deduplicating {len(all_papers_from_csv)} papers found from CSV...")
            # Deduplicate_results_func expects lowercase keys, all_papers_from_csv now has them.
            unique_papers_from_csv = self.deduplicate_results_func(all_papers_from_csv)
            logger.info(f"Found {len(unique_papers_from_csv)} unique papers to save.")

            saved_count = 0
            for paper_to_save in unique_papers_from_csv: # paper_to_save has lowercase keys
                try:
                    # CSVWriterTool's internal validation/mapping handles lowercase input keys like 'title', 'link'
                    self.csv_tool._run(paper_data=paper_to_save)
                    saved_count += 1
                    logger.info(f"Saved paper to main CSV: {paper_to_save.get('title', 'UNKNOWN TITLE')}")
                except Exception as e:
                    logger.error(f"Error saving paper '{paper_to_save.get('title', 'N/A')}' (from input CSV) to main CSV: {e}", exc_info=True)
            logger.info(f"Successfully saved {saved_count} unique papers from the input CSV to '{self.csv_tool.csv_filename if hasattr(self.csv_tool, 'csv_filename') else DEFAULT_CSV_FILE}'.")
        else:
            logger.info("No papers were successfully retrieved from the input CSV to save.")

    def download_pubmed_full_text_from_csv(self, csv_filepath: str = DEFAULT_CSV_FILE):
        logger.info(f"Starting full text download process for PubMed papers in '{csv_filepath}'")
        
        try:
            df = pd.read_csv(csv_filepath, dtype=str, names=self.csv_tool.FIELDNAMES, header=None).fillna('')
            if not df.empty and df.iloc[0][self.csv_tool.FIELDNAMES[0]] == self.csv_tool.FIELDNAMES[0]:
                df = df.iloc[1:].reset_index(drop=True)
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_filepath}. Cannot download full texts.")
            print(f"Error: CSV file '{csv_filepath}' not found.")
            return
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {csv_filepath}. Nothing to process for downloads.")
            print(f"CSV file '{csv_filepath}' is empty.")
            return
        except Exception as e: # Catch other CSV reading errors
            logger.error(f"Error reading or processing CSV '{csv_filepath}' for download: {e}", exc_info=True)
            print(f"Error: Could not process CSV '{csv_filepath}'. Check logs.")
            return

        if 'FullTextPath' not in df.columns:
            df['FullTextPath'] = ''
        df['FullTextPath'] = df['FullTextPath'].astype(str).fillna('')

        original_entrez_email, original_entrez_api_key = Entrez.email, Entrez.api_key
        if self.pubmed_email: Entrez.email = self.pubmed_email
        if self.pubmed_api_key: Entrez.api_key = self.pubmed_api_key

        failed_statuses_to_retry = [
            DOWNLOAD_STATUS_NO_PMCID.lower(), 
            DOWNLOAD_STATUS_NO_PDF_LINK.lower(), 
            DOWNLOAD_STATUS_FAILED.lower(),
            DOWNLOAD_STATUS_HTML_CONTENT.lower(),
            DOWNLOAD_STATUS_PMC_FORBIDDEN.lower(),
            DOWNLOAD_STATUS_PDF_FORBIDDEN.lower(),
            DOWNLOAD_STATUS_PENDING.lower(), # Added PENDING_DOWNLOAD to retry list
            '' # For initial attempts on empty FullTextPath
        ]
        
        papers_to_download_indices = df[
            df['Source'].str.contains('pubmed', case=False, na=False) & 
            (df['FullTextPath'].str.strip().str.lower().isin(failed_statuses_to_retry))
        ].index

        if papers_to_download_indices.empty:
            logger.info("No PubMed papers in CSV require download attempt.")
            print("No PubMed papers in the CSV require a new download attempt.")
            if self.pubmed_email or self.pubmed_api_key: # Restore Entrez settings if they were changed
                Entrez.email, Entrez.api_key = original_entrez_email, original_entrez_api_key
            return

        logger.info(f"Found {len(papers_to_download_indices)} PubMed papers to attempt download.")

        # Create debug_pages directory if it doesn't exist
        DEBUG_PAGES_DIR = "debug_pages"
        if not os.path.exists(DEBUG_PAGES_DIR):
            os.makedirs(DEBUG_PAGES_DIR)

        driver = None # Initialize driver to None, will be set in the try block
        downloaded_count = 0
        
        try:
            # Initialize Selenium WebDriver ONLY if there are papers to download (which we've already checked)
            logger.info("Attempting to initialize Selenium WebDriver...")
            chrome_options = Options()
            chrome_options.add_argument("--headless") # INTENTIONALLY COMMENTED OUT FOR STEP 2
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            chrome_options.add_experimental_option("prefs", {
                "download.default_directory": os.path.abspath(DOWNLOAD_DIR),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "plugins.always_open_pdf_externally": True
            })
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            logger.info("Selenium WebDriver initialized successfully.")

            # ##### START OF ACTUAL DOWNLOAD LOOP #####
            for index in papers_to_download_indices:
                row = df.loc[index]
                pmid_from_csv = ""
                link = row.get('Link', "")

                if "pubmed.ncbi.nlm.nih.gov" in link:
                    match = re.search(r'/(\d+)/?$', link) # Corrected regex for PMID from link
                    if match: pmid_from_csv = match.group(1)
                
                if not pmid_from_csv:
                    logger.warning(f"Could not extract PMID from link '{link}' for row {index}. Setting status to BAD_LINK.")
                    df.loc[index, 'FullTextPath'] = DOWNLOAD_STATUS_BAD_LINK
                    continue

                logger.info(f"Processing PMID: {pmid_from_csv} (Row {index})")
                df.loc[index, 'FullTextPath'] = DOWNLOAD_STATUS_PENDING # Mark as pending

                pmcid = None
                try:
                    handle = Entrez.elink(dbfrom="pubmed", id=pmid_from_csv, linkname="pubmed_pmc")
                    results = Entrez.read(handle)
                    handle.close()
                    if results[0]["LinkSetDb"] and results[0]["LinkSetDb"][0]["Link"]:
                        pmcid = results[0]["LinkSetDb"][0]["Link"][0]["Id"]
                    elif pmid_from_csv == "37633081": # Specific debug log for this PMID
                        logger.warning(f"PMID 37633081: No PMCID direct link found. Raw Entrez results: {results}")
                except Exception as e_entrez:
                    logger.error(f"Entrez error finding PMCID for {pmid_from_csv}: {e_entrez}", exc_info=True)
                    df.loc[index, 'FullTextPath'] = DOWNLOAD_STATUS_NO_PMCID
                    time.sleep(API_CALL_DELAY) # NCBI rate limit
                    continue
                
                if not pmcid:
                    logger.warning(f"No PMCID found for PMID {pmid_from_csv}.")
                    df.loc[index, 'FullTextPath'] = DOWNLOAD_STATUS_NO_PMCID
                    continue

                logger.info(f"Found PMCID: {pmcid} for PMID: {pmid_from_csv}")
                article_page_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
                pdf_url_to_try = None
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

                # Attempt 1: Selenium to find PDF link on article page
                if driver: 
                    try:
                        logger.info(f"Using Selenium to navigate to: {article_page_url}")
                        driver.get(article_page_url)
                        
                        pdf_link_element = None
                        # Updated and reordered selectors
                        selectors_to_try = [
                            # Most specific first, based on user-provided HTML
                            {"by": By.CSS_SELECTOR, "value": "a.usa-link[href^='pdf/'][href$='.pdf']", "condition": EC.element_to_be_clickable, "desc": "Specific USA link starting with pdf/ and ending with .pdf"},
                            # General /pdf/ in href, good chance of matching
                            {"by": By.CSS_SELECTOR, "value": "a[href*='/pdf/'][href$='.pdf']", "condition": EC.element_to_be_clickable, "desc": "Link with /pdf/ in href and ending .pdf"},
                            # Class format-pdf, often used
                            {"by": By.CSS_SELECTOR, "value": "a.format-pdf[href$='.pdf']", "condition": EC.element_to_be_clickable, "desc": "Link with class format-pdf and ending .pdf"},
                            # Data attribute, less common but possible
                            {"by": By.CSS_SELECTOR, "value": "a[data-format='pdf'][href]", "condition": EC.element_to_be_clickable, "desc": "Link with data-format pdf"},
                             # Fallback: any link containing .pdf in href, ensuring it's clickable
                            {"by": By.CSS_SELECTOR, "value": "a[href*='.pdf']", "condition": EC.element_to_be_clickable, "desc": "Any link containing .pdf in href"},
                            # Last resort: visible text PDF, less reliable due to icons
                            {"by": By.PARTIAL_LINK_TEXT, "value": "PDF", "condition": EC.presence_of_element_located, "desc": "Partial link text PDF"}
                        ]

                        for selector_config in selectors_to_try:
                            by_method = selector_config["by"]
                            selector_str = selector_config["value"]
                            condition = selector_config["condition"]
                            desc = selector_config["desc"]
                            try:
                                logger.info(f"Selenium trying selector ({desc}): {by_method} -> '{selector_str}' on {article_page_url}")
                                wait = WebDriverWait(driver, 30) 
                                pdf_link_element = wait.until(condition((by_method, selector_str)))
                                if pdf_link_element and pdf_link_element.get_attribute('href'):
                                    pdf_url_to_try = pdf_link_element.get_attribute('href')
                                    logger.info(f"Selenium found PDF link via ({desc}): {pdf_url_to_try}")
                                    break 
                                else:
                                    logger.info(f"Selector ({desc}) found element but no href or element invalid.")
                                    pdf_link_element = None 
                            except TimeoutException:
                                logger.warning(f"Selenium selector timed out ({desc}): {by_method} -> '{selector_str}' on {article_page_url}")
                            except NoSuchElementException: # Should be caught by TimeoutException with presence_of_element_located
                                logger.warning(f"Selenium selector not found ({desc}): {by_method} -> '{selector_str}' on {article_page_url}")
                            except Exception as e_sel_find:
                                logger.error(f"Selenium error finding link with ({desc}) on {article_page_url}: {e_sel_find}", exc_info=True)
                                pdf_link_element = None
                        
                        if not pdf_url_to_try:
                            logger.error(f"Selenium failed to find PDF link on {article_page_url} after trying all selectors.")
                            try:
                                page_source_filename = os.path.join(DEBUG_PAGES_DIR, f"failed_pmc_{pmcid}_at_{int(time.time())}.html")
                                with open(page_source_filename, "w", encoding="utf-8") as f_ps:
                                    f_ps.write(driver.page_source)
                                logger.info(f"Saved page source for {article_page_url} to {page_source_filename}")
                            except Exception as e_save_ps:
                                logger.error(f"Could not save page source for {article_page_url}: {e_save_ps}")

                    except Exception as e_sel_nav:
                        logger.error(f"General error during Selenium navigation for {article_page_url}: {e_sel_nav}", exc_info=True)
                        # pdf_url_to_try will remain None, will fall through to requests/BeautifulSoup if necessary

                # If Selenium didn't find it or wasn't available, or if the found URL is not absolute
                if not pdf_url_to_try or not pdf_url_to_try.startswith('http'):
                    if pdf_url_to_try: # It means Selenium found something but it wasn't a full URL
                        logger.info(f"Selenium found relative link '{pdf_url_to_try}', attempting to make absolute.")
                        if pdf_url_to_try.startswith('/'):
                            base_url_parts = urlparse(article_page_url)
                            pdf_url_to_try = f"{base_url_parts.scheme}://{base_url_parts.netloc}{pdf_url_to_try}"
                        else: # Try joining with the article page URL as base
                            pdf_url_to_try = urljoin(article_page_url, pdf_url_to_try)
                        logger.info(f"Made absolute: {pdf_url_to_try}")

                    if not (pdf_url_to_try and pdf_url_to_try.startswith('http')): # Still no good URL, try requests+BeautifulSoup
                        logger.info(f"Selenium did not yield a usable PDF URL or driver failed. Falling back to requests & BeautifulSoup for {article_page_url}")
                        try:
                            page_response = requests.get(article_page_url, timeout=30, headers=headers)
                            page_response.raise_for_status()
                            soup = BeautifulSoup(page_response.content, 'html.parser')
                            
                            found_link_tag_bs = None
                            bs_link_patterns = [
                                {'class_': 'format-pdf', 'href': re.compile(r'\.pdf$', re.I)},
                                {'href': re.compile(r'/pdf/.*\.pdf$', re.I)},
                                {'attrs': {'data-format': 'pdf'}, 'href': True},
                                {'string': re.compile(r'pdf', re.I), 'href': True}
                            ]
                            for pattern_kwargs in bs_link_patterns:
                                link_tag = soup.find('a', **pattern_kwargs)
                                if link_tag and link_tag.get('href'):
                                    found_link_tag_bs = link_tag
                                    break
                            
                            if found_link_tag_bs:
                                raw_pdf_url = found_link_tag_bs['href']
                                if raw_pdf_url.startswith('//'): 
                                    pdf_url_to_try = f"https:{raw_pdf_url}"
                                elif raw_pdf_url.startswith('/'):
                                    base_url_parts = urlparse(article_page_url)
                                    pdf_url_to_try = f"{base_url_parts.scheme}://{base_url_parts.netloc}{raw_pdf_url}"
                                elif raw_pdf_url.startswith('http'): 
                                    pdf_url_to_try = raw_pdf_url
                                else: 
                                    pdf_url_to_try = urljoin(article_page_url, raw_pdf_url)
                                logger.info(f"BeautifulSoup found potential PDF link: {pdf_url_to_try}")
                            else:
                                logger.warning(f"BeautifulSoup could not find a clear PDF link on {article_page_url}")

                        except requests.exceptions.HTTPError as e_http:
                            status_to_set = DOWNLOAD_STATUS_PMC_FORBIDDEN if e_http.response.status_code == 403 else DOWNLOAD_STATUS_FAILED
                            logger.error(f"HTTP error ({e_http.response.status_code}) fetching article page {article_page_url} with requests: {e_http}")
                            df.loc[index, 'FullTextPath'] = status_to_set
                            time.sleep(API_CALL_DELAY) # Be respectful
                            continue # Skip to next paper
                        except Exception as e_bs_parse:
                            logger.error(f"Error fetching/parsing article page {article_page_url} with BeautifulSoup: {e_bs_parse}", exc_info=True)
                            # pdf_url_to_try remains None or its previous value from Selenium attempt

                # Attempt 2: Download the PDF if a URL was found (either from Selenium or BeautifulSoup)
                if pdf_url_to_try and pdf_url_to_try.startswith('http'):
                    logger.info(f"Attempting to download PDF from: {pdf_url_to_try}")
                    selenium_download_successful = False

                    if driver: # Try Selenium download first
                        try:
                            logger.info(f"Attempting download via Selenium navigation to: {pdf_url_to_try}")
                            files_before = set(os.listdir(os.path.abspath(DOWNLOAD_DIR)))
                            driver.get(pdf_url_to_try)
                            time.sleep(DOWNLOAD_WAIT_TIME) # Wait for download to likely complete
                            files_after = set(os.listdir(os.path.abspath(DOWNLOAD_DIR)))
                            new_files = files_after - files_before
                            
                            if new_files:
                                downloaded_filename = new_files.pop()
                                if downloaded_filename.lower().endswith('.pdf') and not downloaded_filename.lower().endswith('.crdownload'):
                                    original_filepath = os.path.join(os.path.abspath(DOWNLOAD_DIR), downloaded_filename)
                                    target_filename = f"PMID_{pmid_from_csv}_PMC{pmcid}.pdf"
                                    target_filepath = os.path.join(os.path.abspath(DOWNLOAD_DIR), target_filename)
                                    
                                    # Handle existing target file: remove or rename before new rename
                                    if os.path.exists(target_filepath):
                                        if target_filepath == original_filepath: # Already named correctly, somehow
                                            logger.info(f"Downloaded file '{original_filepath}' is already named correctly.")
                                        else: # Target name exists, but is different from downloaded temp name
                                            logger.warning(f"Target file {target_filepath} already exists. Removing before renaming.")
                                            os.remove(target_filepath)
                                    
                                    if os.path.exists(original_filepath): # Ensure original still exists (wasn't self-renamed)
                                       os.rename(original_filepath, target_filepath)
                                       logger.info(f"Successfully downloaded (via Selenium) and renamed PDF to: {target_filepath}")
                                       df.loc[index, 'FullTextPath'] = target_filepath
                                       downloaded_count += 1
                                       selenium_download_successful = True
                                    else:
                                        # This case might happen if the file was saved directly with the target name
                                        # or if it was already target_filepath and we didn't need to rename.
                                        if os.path.exists(target_filepath):
                                            logger.info(f"File {target_filepath} (target name) exists after Selenium navigation. Assuming successful download.")
                                            df.loc[index, 'FullTextPath'] = target_filepath
                                            downloaded_count += 1
                                            selenium_download_successful = True
                                        else:
                                            logger.warning(f"Original downloaded file '{original_filepath}' disappeared before rename and target '{target_filepath}' not found.")
                                else:
                                    logger.warning(f"New file found via Selenium nav '{downloaded_filename}', but not a PDF or still temp. Will try requests.")
                            else:
                                logger.warning(f"Selenium navigated to {pdf_url_to_try}, but no new file detected in download directory after 15s. Will try requests as fallback.")
                        except Exception as e_sel_download:
                            logger.error(f"Error during Selenium-driven download from {pdf_url_to_try}: {e_sel_download}", exc_info=True)
                            logger.info("Falling back to requests.get() for PDF download.")
                    
                    if not selenium_download_successful:
                        logger.info(f"Attempting download with requests (fallback or direct) from: {pdf_url_to_try}")
                        try:
                            response = requests.get(pdf_url_to_try, timeout=90, stream=True, headers=headers, allow_redirects=True)
                            response.raise_for_status()
                            content_type = response.headers.get('content-type', '').lower()

                            if 'application/pdf' in content_type:
                                file_name = f"PMID_{pmid_from_csv}_PMC{pmcid}.pdf"
                                file_path = os.path.join(DOWNLOAD_DIR, file_name)
                                if os.path.exists(file_path):
                                     logger.warning(f"File {file_path} already exists (requests download). Overwriting.")
                                     # os.remove(file_path) # Optional: remove before writing if overwrite is desired
                                with open(file_path, 'wb') as f:
                                    for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
                                logger.info(f"Successfully downloaded PDF (via requests): {file_path}")
                                df.loc[index, 'FullTextPath'] = file_path
                                downloaded_count += 1
                            elif 'text/html' in content_type:
                                logger.warning(f"URL {pdf_url_to_try} returned HTML content, not PDF. PMCID: {pmcid}.")
                                df.loc[index, 'FullTextPath'] = DOWNLOAD_STATUS_HTML_CONTENT
                                try:
                                    html_error_filename = os.path.join(DEBUG_PAGES_DIR, f"html_instead_of_pdf_pmc_{pmcid}_at_{int(time.time())}.html")
                                    with open(html_error_filename, "w", encoding="utf-8") as f_html_err:
                                        f_html_err.write(response.text)
                                    logger.info(f"Saved unexpected HTML content from {pdf_url_to_try} to {html_error_filename}")
                                except Exception as e_save_html:
                                    logger.error(f"Could not save unexpected HTML content for {pdf_url_to_try}: {e_save_html}")
                            else:
                                logger.warning(f"URL {pdf_url_to_try} returned unexpected content-type: {content_type}. PMCID: {pmcid}")
                                df.loc[index, 'FullTextPath'] = DOWNLOAD_STATUS_FAILED
                        except requests.exceptions.HTTPError as e_pdf_http:
                            status_to_set = DOWNLOAD_STATUS_PDF_FORBIDDEN if e_pdf_http.response.status_code == 403 else DOWNLOAD_STATUS_FAILED
                            logger.error(f"HTTP error ({e_pdf_http.response.status_code}) downloading PDF from {pdf_url_to_try}: {e_pdf_http}")
                            df.loc[index, 'FullTextPath'] = status_to_set
                        except Exception as e_pdf_download_req:
                            logger.error(f"Failed to download PDF (via requests) from {pdf_url_to_try}: {e_pdf_download_req}", exc_info=True)
                            df.loc[index, 'FullTextPath'] = DOWNLOAD_STATUS_FAILED
                
                elif not pdf_url_to_try: # No URL found after all attempts (Selenium and BS)
                    logger.warning(f"No PDF URL could be determined for PMCID {pmcid} after all attempts.")
                    df.loc[index, 'FullTextPath'] = DOWNLOAD_STATUS_NO_PDF_LINK
                else: # URL found but not valid http/https (e.g. relative path that couldn't be absolutized)
                    logger.warning(f"PDF URL '{pdf_url_to_try}' is not a valid absolute HTTP/S URL. PMCID: {pmcid}")
                    df.loc[index, 'FullTextPath'] = DOWNLOAD_STATUS_NO_PDF_LINK # Or a new status like INVALID_PDF_URL
                
                time.sleep(API_CALL_DELAY) # Respect NCBI servers between processing each paper
            ##### END OF ACTUAL DOWNLOAD LOOP #####

        except Exception as e_selenium_or_loop:
            logger.error(f"An error occurred during Selenium initialization or the main download loop: {e_selenium_or_loop}", exc_info=True)
            print(f"Error during download process: {e_selenium_or_loop}. Check logs. Some papers may not have been processed.")
            # If Selenium failed to init, driver will be None. If error in loop, driver might exist.
        
        finally:
            # Save updated CSV regardless of what happened in the try block (unless CSV read failed)
            try:
                # Ensure all FIELDNAMES are present as columns before saving, adding them if missing
                for col_name in self.csv_tool.FIELDNAMES:
                    if col_name not in df.columns:
                        df[col_name] = '' # Add missing column with empty strings
                
                df.to_csv(csv_filepath, index=False, header=True, columns=self.csv_tool.FIELDNAMES, quoting=csv.QUOTE_ALL)
                logger.info(f"Updated CSV saved to {csv_filepath}. Processed/simulated {downloaded_count} papers.")
                print(f"Finished download process. Updated '{csv_filepath}'.")
            except Exception as e_csv_save:
                logger.error(f"Error saving updated CSV to {csv_filepath}: {e_csv_save}", exc_info=True)
                print(f"CRITICAL Error: Could not save updates to '{csv_filepath}'. Check logs.")

            if self.pubmed_email or self.pubmed_api_key: # Restore original Entrez settings
                Entrez.email, Entrez.api_key = original_entrez_email, original_entrez_api_key
            
            if driver: # Quit WebDriver if it was initialized
                try:
                    logger.info("Attempting to quit Selenium WebDriver...")
                    driver.quit()
                    logger.info("Selenium WebDriver quit successfully.")
                except Exception as e_quit:
                    logger.error(f"Error quitting Selenium WebDriver: {e_quit}", exc_info=True)
            else:
                logger.info("Selenium WebDriver was not initialized or failed to initialize, no need to quit.")
                
    # ... (rest of the class)
# ... (main block, display_menu, etc.)

def display_menu():
    print("\n--- PaperPulse Menu ---")
    print("0. Generate Search Queries from Project Description")
    print("1. Scrape New Papers (Manual Queries)")
    print("2. Process Scraped Papers with LLM (Summarize & Validate)")
    print("3. Scrape Papers from Input CSV (using PMID/DOI)")
    print("4. Download Full Text for PubMed Papers in CSV")
    print("5. Exit")
    choice = input("Enter your choice (0-5): ")
    return choice

if __name__ == "__main__":
    load_dotenv() # Load .env file at the start

    # --- Configuration for LLM (if used in Option 2) ---
    # These will be dynamically assigned to the pipeline instance if LLM processing is chosen.
    # os.getenv will now first check actual env vars, then those loaded from .env, then use default.
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini") 
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-1.5-flash-latest")
    DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Configuration for Scraping (always used for Option 1, and PubMed details for pipeline init)
    # These will now be read from .env if present, or fallback to defaults/None
    USER_PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "omarkshoaib@gmail.com") # Default if not in env/.env
    USER_PUBMED_API_KEY = os.getenv("PUBMED_API_KEY") # Will be None if not in env/.env
    
    # Default project description for Option 2 if user doesn't provide one
    DEFAULT_PROJECT_DESC_FOR_LLM = "Evaluate advancements in AI for scientific research automation."
    # --- End Configuration ---

    # Initialize pipeline with details needed for scraping. 
    # LLM provider/model are not passed here to avoid premature LLM setup.
    pipeline = ResearchPipeline(
        pubmed_email=USER_PUBMED_EMAIL,
        pubmed_api_key=USER_PUBMED_API_KEY
    )

    while True:
        user_choice = display_menu()

        if user_choice == '0':
            logger.info("Selected: 0. Generate Search Queries from Project Description")
            project_desc = input("Enter your project description: ").strip()
            if not project_desc:
                logger.warning("Project description cannot be empty.")
                print("Project description cannot be empty. Please try again.")
                continue

            chosen_provider = input(f"Enter LLM provider (gemini, ollama, huggingface) [default: {DEFAULT_LLM_PROVIDER}]: ").strip() or DEFAULT_LLM_PROVIDER
            chosen_model = input(f"Enter LLM model name [default: {DEFAULT_LLM_MODEL}]: ").strip() or DEFAULT_LLM_MODEL
            chosen_base_url = None
            if chosen_provider == "ollama":
                chosen_base_url = input(f"Enter Ollama base URL [default: {DEFAULT_OLLAMA_BASE_URL}]: ").strip() or DEFAULT_OLLAMA_BASE_URL

            pipeline.provider = chosen_provider
            pipeline.model_name = chosen_model
            pipeline.base_url = chosen_base_url
            
            logger.info(f"Attempting to use LLM Provider: {pipeline.provider}, Model: {pipeline.model_name} for query generation.")
            
            try:
                if pipeline.provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
                    raise ValueError("GEMINI_API_KEY environment variable not set.")
                if pipeline.provider == "huggingface" and not os.getenv("HF_TOKEN"):
                    raise ValueError("HF_TOKEN environment variable not set for HuggingFace.")
                pipeline._initialize_llm_resources() 
            except ValueError as ve:
                 logger.error(f"LLM Configuration Error: {ve}")
                 print(f"Error: {ve} Please set it and try again.")
                 continue 
            except ConnectionError as ce:
                 logger.error(f"LLM Connection Error: {ce}")
                 print(f"Error: {ce} Ensure the LLM service is running and accessible.")
                 continue

            try:
                generated_queries = pipeline.generate_search_queries(project_description=project_desc)
                if generated_queries:
                    print("\nGenerated Search Queries:")
                    for i, q_gen in enumerate(generated_queries):
                        print(f"  {i+1}. {q_gen}")
                    
                    print("\nWhat would you like to do with these queries?")
                    print("1. Scrape papers now (raw data to CSV, no LLM processing yet)")
                    print("2. Scrape papers AND then process all pending papers in CSV with LLM")
                    print("3. Return to main menu (do nothing with these queries for now)")
                    action_choice = input("Enter your choice (1-3): ").strip()

                    if action_choice == '1' or action_choice == '2':
                        max_results_str = input(f"Max results per search source for scraping (e.g., 5, default is 3): ")
                        try:
                            max_r = int(max_results_str) if max_results_str.strip() else 3
                            if max_r <= 0: max_r = 3
                        except ValueError:
                            logger.warning("Invalid number for max results. Using default 3.")
                        
                        logger.info(f"Proceeding to scrape with generated queries (max_results: {max_r})...")
                        pipeline.scrape_and_save_raw_papers(search_queries=generated_queries, max_results_per_source=max_r)
                        print("Scraping with generated queries complete.")

                        if action_choice == '2':
                            logger.info("Proceeding to LLM processing for all pending papers in CSV...")
                            print(f"Using project description for validation: '{project_desc[:100]}...'")
                            try:
                                pipeline.process_papers_from_csv(project_description=project_desc, csv_filepath=DEFAULT_CSV_FILE)
                            except RuntimeError as re_err: 
                                 logger.error(f"LLM Runtime error during processing: {re_err}")
                                 print(f"Error: {re_err}")
                            except Exception as e: 
                                logger.error(f"An unexpected error occurred during LLM processing: {e}", exc_info=True)
                                print(f"An unexpected error occurred during LLM processing. Check logs for details.")
                        
                    elif action_choice == '3':
                        print("Queries not used at this time. Returning to main menu.")
                    else:
                        print("Invalid choice for action. Returning to main menu.")
                else:
                    print("Could not generate search queries. Please check the logs, your project description, and LLM setup.")
            except RuntimeError as re_err: 
                 logger.error(f"LLM Runtime error during query generation: {re_err}")
                 print(f"Error: {re_err}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during query generation or subsequent scraping: {e}", exc_info=True)
                print(f"An unexpected error occurred. Check logs for details.")

        elif user_choice == '1':
            logger.info("Selected: 1. Scrape New Papers (Manual Queries)")
            queries_str = input("Enter search queries, separated by commas (e.g., AI in healthcare, LLM for code generation): ")
            if not queries_str.strip():
                logger.warning("No queries entered. Please provide at least one query.")
                print("No queries entered. Please provide at least one query.")
                continue
            queries = [q.strip() for q in queries_str.split(',') if q.strip()] 
            if not queries:
                logger.warning("No valid queries after stripping. Please provide at least one query.")
                print("No valid queries after stripping. Please provide at least one query.")
                continue
            
            max_results_str = input(f"Max results per search source (e.g., 5, default is 3): ")
            try:
                max_r = int(max_results_str) if max_results_str.strip() else 3
                if max_r <= 0: max_r = 3 
            except ValueError:
                logger.warning("Invalid number for max results. Using default 3.")
                max_r = 3
            
            try:
                pipeline.scrape_and_save_raw_papers(search_queries=queries, max_results_per_source=max_r)
            except Exception as e:
                logger.error(f"Error during scraping process: {e}", exc_info=True)
                print(f"An error occurred during scraping. Check logs.")

        elif user_choice == '2':
            logger.info("Selected: 2. Process Scraped Papers with LLM")
            
            chosen_provider = input(f"Enter LLM provider (gemini, ollama, huggingface) [default: {DEFAULT_LLM_PROVIDER}]: ").strip() or DEFAULT_LLM_PROVIDER
            chosen_model = input(f"Enter LLM model name [default: {DEFAULT_LLM_MODEL}]: ").strip() or DEFAULT_LLM_MODEL
            chosen_base_url = None
            if chosen_provider == "ollama":
                chosen_base_url = input(f"Enter Ollama base URL [default: {DEFAULT_OLLAMA_BASE_URL}]: ").strip() or DEFAULT_OLLAMA_BASE_URL

            pipeline.provider = chosen_provider
            pipeline.model_name = chosen_model
            pipeline.base_url = chosen_base_url
            
            logger.info(f"Attempting to use LLM Provider: {pipeline.provider}, Model: {pipeline.model_name} for processing.")
            
            try:
                if pipeline.provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
                     raise ValueError("GEMINI_API_KEY environment variable not set.")
                if pipeline.provider == "huggingface" and not os.getenv("HF_TOKEN"):
                     raise ValueError("HF_TOKEN environment variable not set for HuggingFace.")
                pipeline._initialize_llm_resources()
            except ValueError as ve:
                 logger.error(f"LLM Configuration Error: {ve}")
                 print(f"Error: {ve} Please set it and try again.")
                 continue
            except ConnectionError as ce:
                 logger.error(f"LLM Connection Error: {ce}")
                 print(f"Error: {ce} Ensure the LLM service is running and accessible.")
                 continue

            project_desc_input = input(f"Enter project description for LLM validation (press Enter for default: '{DEFAULT_PROJECT_DESC_FOR_LLM[:60]}...'): ")
            project_desc = project_desc_input.strip() if project_desc_input.strip() else DEFAULT_PROJECT_DESC_FOR_LLM
            
            try:
                pipeline.process_papers_from_csv(project_description=project_desc, csv_filepath=DEFAULT_CSV_FILE)
            except RuntimeError as re_err:
                 logger.error(f"LLM Runtime error during processing: {re_err}")
                 print(f"Error: {re_err}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during LLM processing: {e}", exc_info=True)
                print(f"An unexpected error occurred. Check logs for details.")

        elif user_choice == '3': 
            logger.info("Selected: 3. Scrape Papers from Input CSV (using PMID/DOI)")
            input_csv_file = input("Enter the path to your input CSV file (containing PMID/DOI columns): ").strip()
            if not input_csv_file:
                logger.warning("No input CSV file path provided.")
                print("No input CSV file path provided. Returning to menu.")
                continue
            
            pmid_col_name = input("Enter PMID column name (default: pmid): ").strip().lower() or 'pmid'
            doi_col_name = input("Enter DOI column name (default: doi): ").strip().lower() or 'doi'

            try:
                pipeline.scrape_papers_from_identifiers_csv(input_csv_path=input_csv_file, pmid_col=pmid_col_name, doi_col=doi_col_name)
            except Exception as e:
                logger.error(f"An unexpected error occurred during scraping from input CSV: {e}", exc_info=True)
                print(f"An unexpected error occurred. Check logs for details.")

        elif user_choice == '4':
            logger.info("Selected: 4. Download Full Text for PubMed Papers in CSV")
            try:
                pipeline.download_pubmed_full_text_from_csv(csv_filepath=DEFAULT_CSV_FILE)
            except Exception as e:
                logger.error(f"An unexpected error occurred during full text download: {e}", exc_info=True)
                print(f"An unexpected error occurred during full text download. Check logs.")

        elif user_choice == '5':
            logger.info("Exiting PaperPulse.")
            print("Exiting PaperPulse. Goodbye!")
            break
        else:
            logger.warning(f"Invalid choice: '{user_choice}'. Please enter a number between 0 and 5.")
            print("Invalid choice. Please try again.")

        print("\n") 