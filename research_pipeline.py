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

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Default placeholder for LLM-generated fields
PENDING_LLM_PLACEHOLDER = "PENDING_LLM_PROCESSING"
DEFAULT_CSV_FILE = "research_results.csv"
DOWNLOAD_DIR = "downloaded_papers"
DOWNLOAD_STATUS_NO_PMCID = "NO_PMC_ID_FOUND"
DOWNLOAD_STATUS_NO_PDF_LINK = "NO_PDF_LINK_ON_PMC"
DOWNLOAD_STATUS_FAILED = "DOWNLOAD_FAILED"
DOWNLOAD_STATUS_NOT_PUBMED = "NOT_A_PUBMED_ENTRY"
DOWNLOAD_STATUS_NO_LINK = "NO_LINK_IN_CSV"
DOWNLOAD_STATUS_BAD_LINK = "MALFORMED_PUBMED_LINK"

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
        self.pubmed_email = pubmed_email
        self.pubmed_api_key = pubmed_api_key
        self.arxiv_tool = ArxivSearchTool() 
        # Ensure CSVWriterTool in research_tools.py can accept csv_filename or has a default
        self.csv_tool = CSVWriterTool(csv_filename=DEFAULT_CSV_FILE) 

        try:
            from search_tools import PubMedSearchTool, GoogleScholarSearchTool, deduplicate_results
            self.pubmed_tool = PubMedSearchTool(
                email=self.pubmed_email if self.pubmed_email else os.getenv("PUBMED_EMAIL", "omarkshoaib@gmail.com"),
                api_key=self.pubmed_api_key if self.pubmed_api_key else os.getenv("PUBMED_API_KEY", "e8a81a517e5bf4934d54d0d854b1dcd0b408")
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
            backstory="You are an AI assistant skilled in understanding research topics and formulating precise search queries to retrieve relevant scientific papers. Your response MUST be a single valid JSON array of strings (e.g., [\"query 1\", \"query 2\", \"query 3\"]).",
            verbose=True, allow_delegation=False, llm=self.local_llm
        )
        self.llm_initialized = True
        logger.info("LLM resources initialized successfully.")

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
        crew_result_obj = None
        try:
            crew_result_obj = query_crew.kickoff()
        except Exception as e:
            logger.error(f"Error during query generation crew kickoff: {e}", exc_info=True)
            return [] # Return empty on crew error

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
            summary_result_obj = summary_crew.kickoff()

            densified_abstract_val = "ERROR_SUMMARIZING"
            keywords_str_val = "ERROR_SUMMARIZING"

            if summary_result_obj and hasattr(summary_result_obj, 'raw') and summary_result_obj.raw:
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
            if densified_abstract_val not in ["ERROR_SUMMARIZING", "ERROR_PARSING_SUMMARY", "ERROR_NO_JSON_IN_SUMMARY_OUTPUT", "SKIPPED_EMPTY_ABSTRACT"]:
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
                validation_result_obj = validation_crew.kickoff()
                
                if validation_result_obj and hasattr(validation_result_obj, 'raw') and validation_result_obj.raw:
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
            # Use the FIELDNAMES from CSVWriterTool to define columns, skip the existing header
            df = pd.read_csv(csv_filepath, dtype=str, names=self.csv_tool.FIELDNAMES, header=0).fillna('')
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_filepath}. Cannot download full texts.")
            print(f"Error: CSV file '{csv_filepath}' not found.")
            return
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {csv_filepath}. Nothing to process for downloads.")
            print(f"CSV file '{csv_filepath}' is empty.")
            return

        if 'FullTextPath' not in df.columns:
            logger.info("'FullTextPath' column not found in CSV, adding it.")
            df['FullTextPath'] = ''
        else: # Ensure existing values are treated as strings
            df['FullTextPath'] = df['FullTextPath'].astype(str).fillna('')

        papers_to_download_indices = []
        for index, row in df.iterrows():
            source = str(row.get('Source', '')).strip().lower()
            full_text_path_val = str(row.get('FullTextPath', '')).strip()
            # Only attempt download if it's a PubMed source and path is empty (or not a previous failure/status message)
            if source == 'pubmed' and not full_text_path_val:
                papers_to_download_indices.append(index)
        
        if not papers_to_download_indices:
            logger.info("No PubMed papers found in CSV requiring full text download attempt (or all attempted).")
            return

        logger.info(f"Found {len(papers_to_download_indices)} PubMed papers to attempt full text download.")
        downloaded_count = 0

        for index in papers_to_download_indices:
            row = df.loc[index]
            title = str(row.get('Title', 'N/A'))
            link = str(row.get('Link', '')).strip()
            pmid = None
            
            if not link:
                logger.warning(f"Skipping '{title}' (Row {index+2}): No link found.")
                df.loc[index, 'FullTextPath'] = DOWNLOAD_STATUS_NO_LINK
                continue

            match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/', link)
            if match:
                pmid = match.group(1)
            else:
                logger.warning(f"Skipping '{title}' (Row {index+2}): Could not extract PMID from link: {link}")
                df.loc[index, 'FullTextPath'] = DOWNLOAD_STATUS_BAD_LINK
                continue
            
            logger.info(f"Attempting download for '{title}' (PMID: {pmid}) (Row {index+2})")
            download_status = DOWNLOAD_STATUS_FAILED # Default to failed
            try:
                handle = Entrez.elink(dbfrom="pubmed", id=pmid, linkname="pubmed_pmc_refs")
                record = Entrez.read(handle)
                handle.close()
                
                pmcid = None
                if record[0]["LinkSetDb"]:
                    pmcid_dict = record[0]["LinkSetDb"][0]["Link"][0]
                    pmcid = pmcid_dict['Id']
                
                if pmcid:
                    logger.info(f"Found PMCID: {pmcid} for PMID: {pmid}")
                    # Attempt direct PDF download link pattern (common but not guaranteed)
                    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"
                    
                    # Some PMC articles have slightly different URL structures, this is a common one.
                    # A more robust way involves fetching the article page and parsing for the PDF link.
                    # For now, we try the direct link.
                    try:
                        response = requests.get(pdf_url, timeout=20, stream=True)
                        response.raise_for_status() # Check for HTTP errors
                        
                        # Check content type, ensure it's a PDF
                        content_type = response.headers.get('content-type', '').lower()
                        if 'application/pdf' in content_type:
                            file_name = f"PMID_{pmid}.pdf"
                            file_path = os.path.join(DOWNLOAD_DIR, file_name)
                            with open(file_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            logger.info(f"Successfully downloaded: {file_path}")
                            download_status = file_path
                            downloaded_count += 1
                        else:
                            logger.warning(f"URL for PMCID {pmcid} did not return a PDF. Content-Type: {content_type}. URL: {pdf_url}")
                            download_status = DOWNLOAD_STATUS_NO_PDF_LINK
                    except requests.exceptions.RequestException as req_e:
                        logger.error(f"Failed to download from {pdf_url} for PMCID {pmcid}: {req_e}")
                        download_status = DOWNLOAD_STATUS_FAILED
                else:
                    logger.info(f"No PMCID found for PMID: {pmid}")
                    download_status = DOWNLOAD_STATUS_NO_PMCID
            except Exception as e_entrez:
                logger.error(f"Error fetching PMCID for PMID {pmid}: {e_entrez}")
                download_status = DOWNLOAD_STATUS_FAILED
            finally:
                df.loc[index, 'FullTextPath'] = download_status
                time.sleep(3) # Be polite to NCBI servers
        
        try:
            df.to_csv(csv_filepath, index=False)
            logger.info(f"Full text download attempts complete. Updated '{csv_filepath}'. Successfully downloaded {downloaded_count} files.")
        except Exception as e:
            logger.error(f"Error saving CSV after download attempts: {e}", exc_info=True)

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
    # --- Configuration for LLM (if used in Option 2) ---
    # These will be dynamically assigned to the pipeline instance if LLM processing is chosen.
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini") 
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-1.5-flash-latest")
    DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Configuration for Scraping (always used for Option 1, and PubMed details for pipeline init)
    USER_PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "omarkshoaib@gmail.com")
    USER_PUBMED_API_KEY = os.getenv("PUBMED_API_KEY") 
    
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
                            max_r = 3
                        
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