# PaperPulse 📚

An intelligent research paper scraping and analysis tool that helps researchers stay up-to-date with the latest academic publications across multiple sources.

## Core Functionality

PaperPulse is a command-line tool that offers a multi-step process for managing research papers:

1.  **Query Generation (Optional):**
    *   Users can provide a project description.
    *   An LLM generates a list of relevant search queries.
    *   Users can then choose to proceed with scraping using these queries.

2.  **Multi-Source Scraping:**
    *   Scrapes arXiv, PubMed, and Google Scholar using the generated or manually provided queries.
    *   Can also scrape papers based on a list of PMIDs/DOIs from an input CSV file.
    *   Collects metadata including title, link, authors, year, source, and original abstract.
    *   Deduplicates results based on paper links and titles.

3.  **Raw Data Storage:**
    *   Saves the raw scraped data (including original abstracts and all metadata) to a CSV file (`research_results.csv`).
    *   LLM-dependent fields (densified abstract, keywords, relevance) are initially filled with a placeholder.

4.  **LLM-Powered Processing (Optional):**
    *   Reads the `research_results.csv`.
    *   For papers needing processing:
        *   **Summarization:** Uses an LLM (e.g., Gemini, Ollama, Hugging Face) to create a dense summary of the original abstract (Chain of Density technique).
        *   **Keyword Extraction:** Extracts 3-5 relevant keywords from the abstract.
        *   **Relevance Validation:** Evaluates if the paper is relevant to a given project description based on its title, summary, and keywords, marking it as 'RELEVANT' or 'NOT RELEVANT'.
    *   Updates the CSV file with the LLM-generated content.

5.  **Full-Text Download (PubMed - In Progress):**
    *   Reads `research_results.csv` and attempts to download full-text PDFs for PubMed entries that have a PMCID.
    *   Utilizes `Bio.Entrez` to find PMCIDs from PMIDs.
    *   Employs Selenium to navigate PMC article pages and trigger PDF downloads, aiming to handle JavaScript-based download challenges (e.g., Proof of Work).
    *   Saves downloaded PDFs to the `downloaded_papers` directory and updates the CSV with the file path.
    *   **Current Status:** This feature is under active development. While it can successfully download some papers, it may fail for others due to:
        *   Papers not having a PMCID (i.e., not available in PubMed Central).
        *   Complexities in PDF link detection on some publisher sites or PMC layouts.
        *   Anti-scraping measures or CAPTCHAs that Selenium cannot bypass for all cases.
        *   Inconsistent availability of PDF links even when a PMCID exists.

## Features Summary

- 🔍 **Multi-Source Scraping:** arXiv, Google Scholar, PubMed.
- 📝 **LLM-Powered Summarization:** Chain of Density for concise abstracts.
- 🔑 **Keyword Extraction:** Identifies key terms.
- ✅ **Relevance Validation:** Assesses paper relevance against a project description.
- 📊 **CSV Export:** Manages all data, including LLM outputs and download paths.
- 🤖 **Flexible LLM Support:** Ollama (local), Google Gemini, Hugging Face.
- 📥 **Targeted Scraping:** From user queries or PMID/DOI lists.
- 📄 **Full-Text Download (PubMed):** Attempts to retrieve PDFs for PubMed articles via PMC (experimental, see status above).
- ⚙️ **Terminal Interface:** Menu-driven operation for ease of use.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd PaperPulse
    ```

2.  **Create and activate a conda environment (recommended):**
    ```bash
    conda create -n research_env python=3.9 # Or your preferred Python version
    conda activate research_env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` includes `crewai`, `requests`, `beautifulsoup4`, `pandas`, `biopython`, `selenium`, `webdriver-manager`, `lxml`, `scholarly`)*

4.  **Set up Environment Variables:**
    *   **For Google Gemini:**
        ```bash
        export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        ```
    *   **For Hugging Face:**
        ```bash
        export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
        ```
    *   **For PubMed E-utilities (Optional but Recommended):**
        ```bash
        export PUBMED_EMAIL="your_email@example.com" 
        # export PUBMED_API_KEY="your_pubmed_api_key" # If you have one
        ```
        *If not set, defaults will be used, but providing your email is good practice for NCBI services.*

5.  **Ollama Setup (if using Ollama as LLM provider):**
    *   Install Ollama: [https://ollama.ai/download](https://ollama.ai/download)
    *   Pull your desired model, e.g.:
        ```bash
        ollama pull llama3.1:8b
        ```
    *   Ensure the Ollama server is running when you use it:
        ```bash
        ollama serve
        ```

## Usage

Run the main pipeline script from the terminal:

```bash
python research_pipeline.py
```

This will present a menu with the following options:

**Menu Options:**

*   **0. Generate Search Queries from Project Description:**
    *   Input: Project description, LLM provider/model details.
    *   Output: List of search queries.
    *   Action: Prompts to either scrape with these queries (raw or full process) or return to menu.
*   **1. Scrape New Papers (Manual Queries):**
    *   Input: Comma-separated search queries, max results per source.
    *   Output: Populates `research_results.csv` with raw scraped data.
*   **2. Process Scraped Papers with LLM (Summarize & Validate):**
    *   Input: LLM provider/model details, project description (for validation).
    *   Action: Reads `research_results.csv`, processes entries marked `PENDING_LLM_PROCESSING` for summarization, keyword extraction, and relevance validation. Updates the CSV.
*   **3. Scrape Papers from Input CSV (using PMID/DOI):**
    *   Input: Path to a CSV file containing PMID and/or DOI columns.
    *   Action: Attempts to fetch paper details for each identifier and adds them to `research_results.csv`.
*   **4. Download Full Text for PubMed Papers in CSV:**
    *   Action: Reads `research_results.csv`, attempts to download full-text PDFs for PubMed entries.
    *   Output: Saves PDFs to `downloaded_papers/` and updates `FullTextPath` in the CSV.
    *   *Note: This feature is experimental and may not succeed for all papers.*
*   **5. Exit:**
    *   Terminates the program.

## File Structure

- `research_pipeline.py`: Main executable script with the `ResearchPipeline` class and CLI menu.
- `research_tools.py`: Contains `BaseTool` implementations for arXiv scraping (`ArxivSearchTool`) and CSV writing (`CSVWriterTool`). Defines the `PaperData` Pydantic model.
- `search_tools.py`: Contains `BaseTool` implementations for PubMed (`PubMedSearchTool`) and Google Scholar (`GoogleScholarSearchTool`), and the `deduplicate_results` function.
- `requirements.txt`: Lists project dependencies.
- `research_results.csv`: Default output file for scraped and processed paper data.
- `downloaded_papers/`: Directory where downloaded PDF full texts are stored.
- `debug_pages/`: Directory where HTML source of problematic PMC pages might be saved for debugging downloads.
- `backups/`: Directory where backups of `research_results.csv` are stored before certain operations.

## Contributing

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/your-amazing-feature`).
3.  Commit your changes (`git commit -m 'Add some amazing feature'`).
4.  Push to the branch (`git push origin feature/your-amazing-feature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 