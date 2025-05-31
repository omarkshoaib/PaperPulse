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

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Default placeholder for LLM-generated fields
PENDING_LLM_PLACEHOLDER = "PENDING_LLM_PROCESSING"
DEFAULT_CSV_FILE = "research_results.csv"

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
        
        # Remove LLM-dependent agent initialization from here
        # self.query_agent = ... (This agent will be removed or changed as it was LLM based for query generation)
        # self.summarizer_agent = ... (Will be in _initialize_llm_resources)
        # self.validator_agent = ... (Will be in _initialize_llm_resources)

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
        self.llm_initialized = True
        logger.info("LLM resources initialized successfully.")

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
        expected_cols = ['title', 'link', 'original_abstract', 'authors', 'year', 'source', 'densified_abstract', 'keywords', 'relevance']
        for col in expected_cols:
            if col not in df.columns:
                logger.info(f"Column '{col}' not found in CSV, adding it.")
                df[col] = PENDING_LLM_PLACEHOLDER if col in ['densified_abstract', 'keywords', 'relevance'] else ''
        
        # Ensure original_abstract is not NaN and is a string for LLM processing
        df['original_abstract'] = df['original_abstract'].astype(str).fillna('')
        # Initialize LLM fields if they are completely empty or NaN, to ensure they are strings for comparison
        for col in ['densified_abstract', 'keywords', 'relevance']:
            df[col] = df[col].astype(str).fillna(PENDING_LLM_PLACEHOLDER)
            df[col] = df[col].apply(lambda x: PENDING_LLM_PLACEHOLDER if not x.strip() else x)

        papers_processed_count = 0
        papers_to_update_indices = []

        for index, row in df.iterrows():
            # Check if densified_abstract indicates a need for processing
            current_densified = str(row.get('densified_abstract', '')).strip()
            if current_densified == PENDING_LLM_PLACEHOLDER or not current_densified:
                papers_to_update_indices.append(index)
        
        if not papers_to_update_indices:
            logger.info("No papers found in the CSV that require LLM processing.")
            return

        logger.info(f"Found {len(papers_to_update_indices)} papers to process with LLM.")

        for index in papers_to_update_indices:
            row = df.loc[index]
            title = str(row.get('title', 'N/A'))
            original_abstract_text = str(row.get('original_abstract', '')).strip()

            if not original_abstract_text:
                logger.warning(f"Skipping paper '{title}' due to empty original_abstract.")
                df.loc[index, 'densified_abstract'] = "SKIPPED_EMPTY_ABSTRACT"
                df.loc[index, 'keywords'] = "SKIPPED_EMPTY_ABSTRACT"
                df.loc[index, 'relevance'] = "SKIPPED_EMPTY_ABSTRACT"
                continue

            logger.info(f"Processing paper ({papers_processed_count + 1}/{len(papers_to_update_indices)}): {title}")
            
            # Prepare paper data for the summarizer agent
            paper_for_llm = {
                "title": title,
                "link": str(row.get('link', '')),
                "original_abstract": original_abstract_text,
                "source": str(row.get('source', '')),
                "authors": str(row.get('authors', '')),
                "year": str(row.get('year', ''))
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

            df.loc[index, 'densified_abstract'] = densified_abstract_val
            df.loc[index, 'keywords'] = keywords_str_val
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
            
            df.loc[index, 'relevance'] = relevance_val
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


def display_menu():
    print("\n--- PaperPulse Menu ---")
    print("1. Scrape New Papers (No LLM)")
    print("2. Process Scraped Papers with LLM (Summarize & Validate)")
    print("3. Exit")
    choice = input("Enter your choice (1-3): ")
    return choice

if __name__ == "__main__":
    # --- Configuration for LLM (if used in Option 2) ---
    # These will be dynamically assigned to the pipeline instance if LLM processing is chosen.
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini") 
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-1.5-flash-latest")
    DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Configuration for Scraping (always used for Option 1, and PubMed details for pipeline init)
    USER_PUBMED_EMAIL = os.getenv("PUBMED_EMAIL") # Allow it to be None if not set
    USER_PUBMED_API_KEY = os.getenv("PUBMED_API_KEY") # Allow it to be None if not set
    
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

        if user_choice == '1':
            logger.info("Selected: 1. Scrape New Papers (No LLM)")
            queries_str = input("Enter search queries, separated by commas (e.g., AI in healthcare, LLM for code generation): ")
            if not queries_str.strip():
                logger.warning("No queries entered. Please provide at least one query.")
                continue
            queries = [q.strip() for q in queries_str.split(',') if q.strip()] # Ensure no empty strings
            if not queries:
                logger.warning("No valid queries after stripping. Please provide at least one query.")
                continue
            
            max_results_str = input(f"Max results per search source (e.g., 5, default is 3): ")
            try:
                max_r = int(max_results_str) if max_results_str.strip() else 3
                if max_r <= 0: max_r = 3 # Ensure positive number
            except ValueError:
                logger.warning("Invalid number for max results. Using default 3.")
                max_r = 3
            
            try:
                pipeline.scrape_and_save_raw_papers(search_queries=queries, max_results_per_source=max_r)
            except Exception as e:
                logger.error(f"Error during scraping process: {e}", exc_info=True)

        elif user_choice == '2':
            logger.info("Selected: 2. Process Scraped Papers with LLM")
            
            # Get LLM configuration for this run
            chosen_provider = input(f"Enter LLM provider (e.g., gemini, ollama, huggingface) [default: {DEFAULT_LLM_PROVIDER}]: ").strip() or DEFAULT_LLM_PROVIDER
            chosen_model = input(f"Enter LLM model name (e.g., gemini-1.5-flash-latest, llama3:8b) [default: {DEFAULT_LLM_MODEL}]: ").strip() or DEFAULT_LLM_MODEL
            chosen_base_url = None
            if chosen_provider == "ollama":
                chosen_base_url = input(f"Enter Ollama base URL [default: {DEFAULT_OLLAMA_BASE_URL}]: ").strip() or DEFAULT_OLLAMA_BASE_URL

            # Assign LLM config to the pipeline instance for this operation
            pipeline.provider = chosen_provider
            pipeline.model_name = chosen_model
            pipeline.base_url = chosen_base_url
            
            logger.info(f"Attempting to use LLM Provider: {pipeline.provider}, Model: {pipeline.model_name}")
            if pipeline.provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
                logger.error("GEMINI_API_KEY environment variable not set. Cannot use Gemini for processing.")
                print("Error: GEMINI_API_KEY not set. Please set it in your environment to use Gemini.")
                continue
            if pipeline.provider == "huggingface" and not os.getenv("HF_TOKEN"):
                logger.error("HF_TOKEN environment variable not set. Cannot use Hugging Face for processing.")
                print("Error: HF_TOKEN not set. Please set it in your environment to use Hugging Face.")
                continue

            project_desc_input = input(f"Enter project description for LLM validation (press Enter for default: '{DEFAULT_PROJECT_DESC_FOR_LLM[:60]}...'): ")
            project_desc = project_desc_input.strip() if project_desc_input.strip() else DEFAULT_PROJECT_DESC_FOR_LLM
            
            try:
                pipeline.process_papers_from_csv(project_description=project_desc, csv_filepath=DEFAULT_CSV_FILE)
            except ValueError as ve: # Catch ValueError from _initialize_llm_resources
                 logger.error(f"Configuration error for LLM processing: {ve}")
                 print(f"Error: {ve}")
            except ConnectionError as ce:
                 logger.error(f"Connection error during LLM processing: {ce}")
                 print(f"Error: {ce}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during LLM processing: {e}", exc_info=True)
                print(f"An unexpected error occurred. Check logs for details.")

        elif user_choice == '3':
            logger.info("Exiting PaperPulse.")
            print("Exiting PaperPulse. Goodbye!")
            break
        else:
            logger.warning("Invalid choice. Please enter a number between 1 and 3.")
            print("Invalid choice. Please try again.")

        print("\n") # Newline for readability before showing menu again