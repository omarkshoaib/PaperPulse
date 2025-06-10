import gradio as gr
import pandas as pd
import os
import json
import time
from datetime import datetime
from research_pipeline import ResearchPipeline, DEFAULT_CSV_FILE

# --- Configuration ---
# Load configurations from environment or .env file, similar to research_pipeline.py
# This ensures that the Gradio app uses the same API keys and settings.
from dotenv import load_dotenv
load_dotenv()

DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-1.5-flash-latest")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
USER_PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "omarkshoaib@gmail.com")
USER_PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")

# --- Initialize the Research Pipeline ---
# We initialize it once to be used by all interface functions.
pipeline = ResearchPipeline(
    pubmed_email=USER_PUBMED_EMAIL,
    pubmed_api_key=USER_PUBMED_API_KEY
)

# --- Helper Functions ---

def display_csv_content(csv_file=DEFAULT_CSV_FILE):
    """Helper function to read and display the current CSV file."""
    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading CSV for display: {e}")
        return pd.DataFrame()

def get_csv_stats(csv_file=DEFAULT_CSV_FILE):
    """Get statistics about the CSV file."""
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            return "No papers in database."
        
        total_papers = len(df)
        source_counts = df['Source'].value_counts().to_dict() if 'Source' in df.columns else {}
        
        # Count processed vs pending papers
        processed_count = 0
        pending_count = 0
        if 'Densified Abstract' in df.columns:
            processed_count = len(df[df['Densified Abstract'].notna() & 
                                   (df['Densified Abstract'] != 'PENDING_LLM_PROCESSING')])
            pending_count = total_papers - processed_count
        
        stats_text = f"""
📊 **Database Statistics:**
- **Total Papers**: {total_papers}
- **Processed by LLM**: {processed_count}
- **Pending Processing**: {pending_count}

📚 **Papers by Source:**
"""
        for source, count in source_counts.items():
            stats_text += f"- {source}: {count}\n"
            
        return stats_text
    except Exception as e:
        return f"Error reading CSV stats: {e}"

def export_csv_filtered(df_filtered):
    """Export filtered CSV data."""
    if df_filtered is None or df_filtered.empty:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"paperpulse_export_{timestamp}.csv"
    export_path = os.path.join(os.getcwd(), export_filename)
    
    df_filtered.to_csv(export_path, index=False)
    return export_path

# --- Workflow Functions ---

def generate_queries_only(project_desc, llm_provider, llm_model, ollama_url):
    """Generate search queries without scraping."""
    yield "Initializing LLM resources...", gr.skip()
    try:
        pipeline.provider = llm_provider
        pipeline.model_name = llm_model
        pipeline.base_url = ollama_url
        pipeline._initialize_llm_resources()
    except Exception as e:
        yield f"❌ Error initializing LLM: {e}", gr.skip()
        return

    yield "🔍 Generating search queries...", gr.skip()
    try:
        queries = pipeline.generate_search_queries(project_description=project_desc)
        if queries:
            queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
            yield f"✅ **Generated {len(queries)} Search Queries:**\n\n{queries_text}", queries
        else:
            yield "❌ Failed to generate search queries. Please check your project description and LLM setup.", []
    except Exception as e:
        yield f"❌ Error during query generation: {e}", []

def scrape_papers_only(queries_text, max_results, source_arxiv, source_pubmed, source_scholar):
    """Scrape papers using provided queries."""
    if not queries_text:
        yield "❌ No queries provided. Please generate or enter queries first.", gr.skip()
        return
    
    # Parse queries from different input formats
    queries = []
    
    # If it's a list (from JSON), use directly
    if isinstance(queries_text, list):
        queries = [str(q).strip() for q in queries_text if str(q).strip()]
    else:
        # Convert to string if needed
        queries_str = str(queries_text).strip()
        
        # Try to parse as JSON first
        try:
            if queries_str.startswith('[') and queries_str.endswith(']'):
                parsed_queries = json.loads(queries_str)
                if isinstance(parsed_queries, list):
                    queries = [str(q).strip() for q in parsed_queries if str(q).strip()]
                else:
                    queries = [str(parsed_queries).strip()]
            else:
                # Parse as text (line-separated or comma-separated)
                queries = [q.strip() for q in queries_str.replace('\n', ',').split(',') if q.strip()]
        except (json.JSONDecodeError, ValueError):
            # Fallback to text parsing
            queries = [q.strip() for q in queries_str.replace('\n', ',').split(',') if q.strip()]
    
    if not queries:
        yield "❌ Could not parse queries. Please check the format.", gr.skip()
        return
    
    # Filter sources based on user selection
    active_sources = []
    if source_arxiv: active_sources.append("arXiv")
    if source_pubmed: active_sources.append("PubMed") 
    if source_scholar: active_sources.append("Google Scholar")
    
    if not active_sources:
        yield "❌ Please select at least one source to scrape from.", gr.skip()
        return
    
    yield f"🚀 Starting scrape from {', '.join(active_sources)} with {len(queries)} queries...", gr.skip()
    
    try:
        # Temporarily modify the pipeline to only use selected sources
        original_tools = {}
        if not source_arxiv:
            original_tools['arxiv'] = pipeline.arxiv_tool
            pipeline.arxiv_tool = None
        if not source_pubmed:
            original_tools['pubmed'] = pipeline.pubmed_tool
            pipeline.pubmed_tool = None
        if not source_scholar:
            original_tools['scholar'] = pipeline.google_scholar_tool
            pipeline.google_scholar_tool = None
        
        pipeline.scrape_and_save_raw_papers(search_queries=queries, max_results_per_source=int(max_results))
        
        # Restore original tools
        for tool_name, tool in original_tools.items():
            if tool_name == 'arxiv':
                pipeline.arxiv_tool = tool
            elif tool_name == 'pubmed':
                pipeline.pubmed_tool = tool
            elif tool_name == 'scholar':
                pipeline.google_scholar_tool = tool
        
        results_df = display_csv_content()
        yield f"✅ Scraping complete! Results saved to {DEFAULT_CSV_FILE}.", results_df
        
    except Exception as e:
        yield f"❌ Error during scraping: {e}", gr.skip()

def manual_scrape_workflow(queries_text, max_results, source_arxiv, source_pubmed, source_scholar):
    """Manual scraping workflow with user-provided queries."""
    yield from scrape_papers_only(queries_text, max_results, source_arxiv, source_pubmed, source_scholar)

def process_csv_workflow(project_desc, llm_provider, llm_model, ollama_url):
    """Process existing CSV with LLM."""
    yield "🤖 Initializing LLM resources for processing...", gr.skip()
    try:
        pipeline.provider = llm_provider
        pipeline.model_name = llm_model
        pipeline.base_url = ollama_url
        pipeline._initialize_llm_resources()
    except Exception as e:
        yield f"❌ Error initializing LLM: {e}", gr.skip()
        return

    yield f"📝 Processing papers in {DEFAULT_CSV_FILE}...", gr.skip()
    try:
        pipeline.process_papers_from_csv(project_description=project_desc, csv_filepath=DEFAULT_CSV_FILE)
        results_df = display_csv_content()
        yield f"✅ LLM processing complete! {DEFAULT_CSV_FILE} has been updated.", results_df
    except Exception as e:
        yield f"❌ Error during LLM processing: {e}", gr.skip()

def scrape_from_identifiers_workflow(upload_file, pmid_col, doi_col):
    """Scrape papers from CSV with identifiers."""
    if upload_file is None:
        yield "❌ Please upload a CSV file.", gr.skip()
        return

    input_csv_path = upload_file.name
    yield f"📥 Starting scrape from identifiers in {os.path.basename(input_csv_path)}...", gr.skip()
    try:
        pipeline.scrape_papers_from_identifiers_csv(
            input_csv_path=input_csv_path,
            pmid_col=pmid_col,
            doi_col=doi_col
        )
        updated_df = display_csv_content()
        yield "✅ Scraping from identifiers complete!", updated_df
    except Exception as e:
        yield f"❌ Error during identifier scraping: {e}", gr.skip()

def download_full_text_workflow():
    """Download full-text PDFs."""
    yield f"📄 Starting full-text download for PubMed papers in {DEFAULT_CSV_FILE}...", gr.skip()
    try:
        pipeline.download_pubmed_full_text_from_csv(csv_filepath=DEFAULT_CSV_FILE)
        updated_df = display_csv_content()
        yield "✅ Full-text download process complete! Check the 'downloaded_papers' directory.", updated_df
    except Exception as e:
        yield f"❌ Error during download: {e}", gr.skip()

def filter_csv_data(df, source_filter, relevance_filter, search_term):
    """Filter CSV data based on user criteria."""
    if df is None or df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Source filter
    if source_filter and source_filter != "All":
        filtered_df = filtered_df[filtered_df['Source'] == source_filter]
    
    # Relevance filter  
    if relevance_filter and relevance_filter != "All" and 'Relevance' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Relevance'] == relevance_filter]
    
    # Search term filter
    if search_term:
        search_cols = ['Title', 'Authors', 'Original Abstract', 'Keywords']
        search_cols = [col for col in search_cols if col in filtered_df.columns]
        
        mask = filtered_df[search_cols].astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        filtered_df = filtered_df[mask]
    
    return filtered_df

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft(), title="PaperPulse - Research Automation") as demo:
    
    # Header
    gr.Markdown("# 🔬 PaperPulse: Research Automation Interface")
    gr.Markdown("*Streamline your academic literature discovery and analysis*")
    
    with gr.Tabs():
        
        # 📊 Dashboard Tab
        with gr.TabItem("📊 Dashboard"):
            gr.Markdown("## Overview & Quick Stats")
            
            with gr.Row():
                with gr.Column(scale=2):
                    csv_stats = gr.Markdown(value=get_csv_stats(), label="Database Statistics")
                    refresh_stats_btn = gr.Button("🔄 Refresh Stats", size="sm")
                    
                with gr.Column(scale=3):
                    current_csv_preview = gr.DataFrame(
                        value=display_csv_content(),
                        label="Current Database Preview",
                        interactive=False
                    )
            
            refresh_stats_btn.click(
                fn=lambda: (get_csv_stats(), display_csv_content()),
                outputs=[csv_stats, current_csv_preview]
            )
        
        # 🔍 Query Generation Tab  
        with gr.TabItem("🔍 Generate Queries"):
            gr.Markdown("## AI-Powered Query Generation")
            gr.Markdown("Describe your research project and let AI generate optimized search queries.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    project_description_input = gr.Textbox(
                        lines=6, 
                        label="Project Description", 
                        placeholder="Enter a detailed description of your research topic, goals, and key concepts..."
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### LLM Configuration")
                    llm_provider_input_gen = gr.Dropdown(
                        ["gemini", "ollama", "huggingface"], 
                        value=DEFAULT_LLM_PROVIDER, 
                        label="LLM Provider"
                    )
                    llm_model_input_gen = gr.Textbox(value=DEFAULT_LLM_MODEL, label="LLM Model Name")
                    ollama_url_input_gen = gr.Textbox(value=DEFAULT_OLLAMA_BASE_URL, label="Ollama Base URL (if applicable)")
            
            generate_btn = gr.Button("🎯 Generate Search Queries", variant="primary", size="lg")
            
            with gr.Row():
                gen_status_output = gr.Textbox(label="Generation Status", interactive=False, lines=3)
                generated_queries_output = gr.JSON(label="Generated Queries", visible=True)
            
            generate_btn.click(
                fn=generate_queries_only,
                inputs=[project_description_input, llm_provider_input_gen, llm_model_input_gen, ollama_url_input_gen],
                outputs=[gen_status_output, generated_queries_output]
            )
        
        # 📥 Manual Query & Scraping Tab
        with gr.TabItem("📥 Manual Scraping"):
            gr.Markdown("## Manual Query Input & Scraping")
            gr.Markdown("Enter your own search queries and scrape papers from selected sources.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    manual_queries_input = gr.Textbox(
                        lines=8,
                        label="Search Queries (one per line or comma-separated)",
                        placeholder="Enter your search queries here...\nExample:\nGraph Convolutional Networks action recognition\nGCN interpretability multimodal\nSkeleton-based action recognition"
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Scraping Configuration")
                    max_results_manual = gr.Slider(
                        minimum=1, maximum=50, value=10, step=1, 
                        label="Max Results per Source"
                    )
                    
                    gr.Markdown("**Select Sources:**")
                    source_arxiv = gr.Checkbox(label="📚 arXiv", value=True)
                    source_pubmed = gr.Checkbox(label="🏥 PubMed", value=True)  
                    source_scholar = gr.Checkbox(label="🎓 Google Scholar", value=True)
            
            scrape_manual_btn = gr.Button("🚀 Start Scraping", variant="primary", size="lg")
            
            with gr.Row():
                manual_status_output = gr.Textbox(label="Scraping Status", interactive=False, lines=3)
                manual_results_output = gr.DataFrame(label="Scraped Papers", interactive=False)
            
            scrape_manual_btn.click(
                fn=manual_scrape_workflow,
                inputs=[manual_queries_input, max_results_manual, source_arxiv, source_pubmed, source_scholar],
                outputs=[manual_status_output, manual_results_output]
            )
        
        # 🤖 LLM Processing Tab
        with gr.TabItem("🤖 LLM Processing"):
            gr.Markdown("## Summarize & Validate Papers")
            gr.Markdown(f"Process papers in **{DEFAULT_CSV_FILE}** with AI to generate summaries, keywords, and relevance scores.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    project_desc_process = gr.Textbox(
                        lines=6, 
                        label="Project Description for Validation", 
                        placeholder="Enter the same project description used for scraping to ensure consistent validation..."
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### LLM Configuration")
                    llm_provider_input_proc = gr.Dropdown(
                        ["gemini", "ollama", "huggingface"], 
                        value=DEFAULT_LLM_PROVIDER, 
                        label="LLM Provider"
                    )
                    llm_model_input_proc = gr.Textbox(value=DEFAULT_LLM_MODEL, label="LLM Model Name")
                    ollama_url_input_proc = gr.Textbox(value=DEFAULT_OLLAMA_BASE_URL, label="Ollama Base URL (if applicable)")

            start_processing_button = gr.Button("🤖 Start LLM Processing", variant="primary", size="lg")
            
            with gr.Row():
                processing_status_output = gr.Textbox(label="Processing Status", interactive=False, lines=3)
                processing_results_output = gr.DataFrame(label="Updated Papers", interactive=False)

            start_processing_button.click(
                fn=process_csv_workflow,
                inputs=[project_desc_process, llm_provider_input_proc, llm_model_input_proc, ollama_url_input_proc],
                outputs=[processing_status_output, processing_results_output]
            )
        
        # 📚 Library Management Tab
        with gr.TabItem("📚 Library Manager"):
            gr.Markdown("## Browse, Filter & Export Papers")
            gr.Markdown("Manage your paper database with advanced filtering and export capabilities.")
            
            # Filters
            with gr.Row():
                source_filter = gr.Dropdown(
                    ["All", "arXiv", "PubMed", "Google Scholar"], 
                    value="All", 
                    label="Filter by Source"
                )
                relevance_filter = gr.Dropdown(
                    ["All", "RELEVANT", "NOT RELEVANT"], 
                    value="All", 
                    label="Filter by Relevance"
                )
                search_term = gr.Textbox(
                    label="Search in Title/Authors/Abstract", 
                    placeholder="Enter search term..."
                )
            
            with gr.Row():
                filter_btn = gr.Button("🔍 Apply Filters", variant="secondary")
                export_btn = gr.Button("📁 Export Filtered Results", variant="primary")
                clear_filters_btn = gr.Button("🧹 Clear Filters", variant="secondary")
            
            # Results display
            filtered_results = gr.DataFrame(
                value=display_csv_content(),
                label="Papers Database",
                interactive=False,
                wrap=True
            )
            
            export_file = gr.File(label="Download Exported CSV", visible=False)
            
            def apply_filters(source, relevance, search):
                df = display_csv_content()
                filtered_df = filter_csv_data(df, source, relevance, search)
                return filtered_df
            
            def export_filtered_data(source, relevance, search):
                df = display_csv_content()
                filtered_df = filter_csv_data(df, source, relevance, search)
                export_path = export_csv_filtered(filtered_df)
                return export_path, gr.update(visible=True)
            
            def clear_all_filters():
                return "All", "All", "", display_csv_content()
            
            filter_btn.click(
                fn=apply_filters,
                inputs=[source_filter, relevance_filter, search_term],
                outputs=[filtered_results]
            )
            
            export_btn.click(
                fn=export_filtered_data,
                inputs=[source_filter, relevance_filter, search_term],
                outputs=[export_file, export_file]
            )
            
            clear_filters_btn.click(
                fn=clear_all_filters,
                outputs=[source_filter, relevance_filter, search_term, filtered_results]
            )
        
        # 📄 Full Text Tab
        with gr.TabItem("📄 Full Text"):
            gr.Markdown("## Download & Manage PDFs")
            gr.Markdown("Download full-text PDFs and manage your document library.")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Download PDFs")
                    gr.Markdown("Download full-text PDFs for PubMed papers in your database.")
                    
                    start_download_button = gr.Button("📄 Start Downloading PubMed PDFs", variant="primary", size="lg")
                    download_status_output = gr.Textbox(label="Download Status", interactive=False, lines=3)
                    
                with gr.Column():
                    gr.Markdown("### Document Browser")
                    gr.Markdown("Browse downloaded papers in the `downloaded_papers/` directory.")
                    
                    # Simple file browser (list downloaded files)
                    def list_downloaded_files():
                        download_dir = "downloaded_papers"
                        if os.path.exists(download_dir):
                            files = [f for f in os.listdir(download_dir) if f.endswith('.pdf')]
                            return f"📁 **Downloaded Files ({len(files)}):**\n" + "\n".join([f"• {f}" for f in files])
                        return "📁 No downloaded files found."
                    
                    file_list = gr.Markdown(value=list_downloaded_files())
                    refresh_files_btn = gr.Button("🔄 Refresh File List", size="sm")
                    
                    refresh_files_btn.click(fn=list_downloaded_files, outputs=[file_list])

            download_results_output = gr.DataFrame(label="Updated CSV", interactive=False)

            start_download_button.click(
                fn=download_full_text_workflow,
                outputs=[download_status_output, download_results_output]
            )
        
        # 📥 Import from Identifiers Tab
        with gr.TabItem("📥 Import Identifiers"):
            gr.Markdown("## Import from PMID/DOI CSV")
            gr.Markdown("Upload a CSV file with PubMed IDs or DOIs to fetch paper details.")
            
            with gr.Row():
                with gr.Column():
                    input_csv_upload = gr.File(label="📁 Upload Identifier CSV", file_types=[".csv"])
                    pmid_col_input = gr.Textbox(value="pmid", label="PMID Column Name")
                    doi_col_input = gr.Textbox(value="doi", label="DOI Column Name")
            
            start_id_scrape_button = gr.Button("📥 Import from CSV", variant="primary", size="lg")
            
            with gr.Row():
                id_scrape_status_output = gr.Textbox(label="Import Status", interactive=False, lines=3)
                id_scrape_results_output = gr.DataFrame(label="Updated Database", interactive=False)

            start_id_scrape_button.click(
                fn=scrape_from_identifiers_workflow,
                inputs=[input_csv_upload, pmid_col_input, doi_col_input],
                outputs=[id_scrape_status_output, id_scrape_results_output]
            )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860) 