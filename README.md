# 🔬 PaperPulse: AI-Powered Research Automation

> **Streamline your academic literature discovery and analysis with intelligent automation**

PaperPulse is a powerful research automation tool that helps researchers efficiently discover, collect, process, and analyze academic papers from multiple sources using AI-powered workflows.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/Interface-Gradio-orange.svg)](https://gradio.app)

## 🌟 Key Features

### 🔍 **Multi-Source Paper Discovery**
- **arXiv**: Preprints and open-access papers
- **PubMed**: Biomedical and life science literature
- **Google Scholar**: Broad academic coverage across disciplines

### 🤖 **AI-Powered Processing**
- **Smart Query Generation**: AI creates optimized search queries from project descriptions
- **Intelligent Summarization**: Chain of Density technique for concise, information-rich abstracts
- **Keyword Extraction**: Automatic identification of 3-5 most relevant keywords
- **Relevance Assessment**: AI determines paper relevance to your research project

### 🖥️ **Dual Interface Options**
- **🌐 Web Interface**: Modern, user-friendly Gradio-based GUI
- **💻 Terminal Interface**: Command-line menu for advanced users

### 📊 **Advanced Data Management**
- **Interactive Filtering**: Sort and filter by source, relevance, keywords
- **Export Capabilities**: Download filtered results as CSV
- **Real-time Statistics**: Track collection progress and database stats
- **PDF Management**: Automatic full-text download for supported papers

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone the repository
git clone https://github.com/your-username/PaperPulse.git
cd PaperPulse

# Create and activate conda environment (recommended)
conda create -n research_env python=3.9
conda activate research_env

# Install dependencies
pip install -r requirements.txt
```

### 2. **Configuration**

Create a `.env` file in the project root with your API keys:

```env
# Required for AI features
GEMINI_API_KEY=your_gemini_api_key_here

# Required for PubMed access
PUBMED_EMAIL=your_email@example.com
PUBMED_API_KEY=your_pubmed_api_key_here

# Optional: Rate limiting (default: 90 seconds)
API_CALL_DELAY=90
DOWNLOAD_WAIT_TIME=90

# Optional: LLM provider defaults
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-1.5-flash-latest
```

### 3. **Launch the Interface**

#### 🌐 **Web Interface** (Recommended)
```bash
python app.py
```
Then open http://127.0.0.1:7860 in your browser

#### 💻 **Terminal Interface**
```bash
python research_pipeline.py
```

## 📖 Complete Usage Guide

### 🌐 **Web Interface Workflow**

#### **Step 1: Dashboard Overview**
- View database statistics and recent papers
- Monitor collection progress
- Quick access to all features

#### **Step 2: Generate Search Queries**
1. Navigate to **🔍 Generate Queries** tab
2. Enter your research project description
3. Configure LLM settings (Gemini, Ollama, or HuggingFace)
4. Click **Generate Search Queries**
5. Review and copy generated queries

#### **Step 3: Collect Papers**
1. Go to **📥 Manual Scraping** tab
2. Paste generated queries or enter your own
3. Select sources (arXiv, PubMed, Google Scholar)
4. Set max results per source
5. Click **Start Scraping**

#### **Step 4: AI Processing**
1. Navigate to **🤖 LLM Processing** tab
2. Enter your project description for validation
3. Configure LLM settings
4. Click **Start LLM Processing**

#### **Step 5: Manage Your Library**
1. Use **📚 Library Manager** tab to:
   - Filter by source, relevance, keywords
   - Search within titles, authors, abstracts
   - Export filtered results as CSV
   - Browse and organize your collection

#### **Step 6: Download Full Texts**
1. Go to **📄 Full Text** tab
2. Click **Start Downloading PubMed PDFs**
3. Monitor download progress
4. Access files in the **Document Browser**

### 💻 **Terminal Interface Options**

```
--- PaperPulse Menu ---
0. Generate Search Queries from Project Description
1. Scrape New Papers (Manual Queries)
2. Process Scraped Papers with LLM (Summarize & Validate)
3. Scrape Papers from Input CSV (using PMID/DOI)
4. Download Full Text for PubMed Papers in CSV
5. Exit
```

## 🔧 Advanced Configuration

### **LLM Providers Setup**

#### **Google Gemini** (Recommended)
```bash
# Get API key from: https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your_api_key_here"
```

#### **Ollama** (Local LLM)
```bash
# Install Ollama: https://ollama.ai/download
ollama pull llama3.1:8b  # or your preferred model
ollama serve  # Start the server
```

#### **HuggingFace**
```bash
# Get token from: https://huggingface.co/settings/tokens
export HF_TOKEN="your_token_here"
```

### **PubMed API Setup**
1. Register at: https://www.ncbi.nlm.nih.gov/account/
2. Get API key from: https://www.ncbi.nlm.nih.gov/account/settings/
3. Add to your `.env` file

### **Rate Limiting Configuration**
```env
# Adjust based on your API limits and needs
API_CALL_DELAY=90          # Seconds between API calls
DOWNLOAD_WAIT_TIME=90      # Seconds for download operations
LLM_CALL_DELAY=5          # Seconds between LLM calls
LLM_RETRY_DELAY=30        # Retry delay after rate limits
LLM_MAX_RETRIES=3         # Max retry attempts
```

## 📁 Project Structure

```
PaperPulse/
├── app.py                 # Gradio web interface
├── research_pipeline.py   # Terminal interface & core logic
├── research_tools.py      # arXiv scraping & CSV tools
├── search_tools.py        # PubMed & Google Scholar tools
├── .env                   # Configuration file (create this)
├── requirements.txt       # Python dependencies
├── GCN_DAM_2.csv         # Default output database
├── downloaded_papers/     # PDF storage directory
└── README.md             # This file
```

## 📊 Output Files

### **Main Database CSV**
- **Filename**: `GCN_DAM_2.csv` (or custom name)
- **Columns**: Title, Link, Authors, Year, Source, Original Abstract, Densified Abstract, Keywords, Relevance, Timestamp, FullTextPath

### **Downloaded PDFs**
- **Location**: `downloaded_papers/` directory
- **Naming**: Organized by paper title and source
- **Formats**: PDF files with full-text content

### **Export Files**
- **Location**: Project root directory
- **Format**: `paperpulse_export_YYYYMMDD_HHMMSS.csv`
- **Content**: Filtered results based on your criteria

## 🔍 Example Research Workflows

### **Academic Literature Review**
1. **Generate queries** from your research proposal
2. **Scrape all sources** for comprehensive coverage
3. **AI processing** to identify most relevant papers
4. **Filter by relevance** and export for analysis
5. **Download full texts** for detailed review

### **Trend Analysis**
1. **Manual queries** for specific topics over time
2. **Filter by publication year** and source
3. **Keyword analysis** to identify emerging themes
4. **Export data** for statistical analysis

### **Citation Building**
1. **Import from PMID/DOI lists** from other studies
2. **AI summarization** for quick understanding
3. **Relevance validation** against your work
4. **Export formatted** for bibliography management

## 🛠️ Troubleshooting

### **Common Issues**

#### **"API Key Not Found"**
- Ensure your `.env` file is in the project root
- Check that API keys are correctly formatted
- Restart the application after adding keys

#### **"No Results Found"**
- Try simpler, more general search terms
- Check if the specific database has content for your topic
- Verify your internet connection

#### **"LLM Processing Failed"**
- Check your LLM provider's rate limits
- Ensure sufficient API credits/quota
- Try reducing batch sizes or adding delays

#### **"PDF Download Failed"**
- Not all papers have freely available PDFs
- Some publishers block automated downloads
- PubMed Central availability varies by paper

### **Performance Tips**
- Use specific keywords for better results
- Start with smaller result limits (5-10 per source)
- Process papers in batches for large collections
- Regular backups of your CSV database

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Development Setup**
```bash
# Clone your fork
git clone https://github.com/your-username/PaperPulse.git
cd PaperPulse

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8  # Optional: testing and formatting tools

# Run tests (if available)
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CrewAI** for agent orchestration
- **Gradio** for the web interface framework
- **Scholarly** for Google Scholar access
- **BioPython** for PubMed integration
- **LiteLLM** for unified LLM provider interface

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/PaperPulse/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/PaperPulse/discussions)
- **Email**: your-email@example.com

---

**🔬 Happy Researching with PaperPulse!**

*Automate your literature discovery and focus on what matters most - your research.* 