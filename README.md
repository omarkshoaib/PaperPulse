# PaperPulse 📚

An intelligent research paper scraping and analysis tool that helps researchers stay up-to-date with the latest academic publications across multiple sources.

## Features

- 🔍 Multi-source paper scraping (arXiv, Google Scholar, PubMed)
- 📝 Intelligent paper summarization using Chain of Density
- ✅ Relevance validation using LLMs
- 📊 CSV export with relevance tracking
- 🤖 Support for multiple LLM providers (Ollama, Gemini, Hugging Face)

## Current Status

### Main Branch
- ✅ arXiv paper scraping
- ✅ Paper summarization
- ✅ Relevance validation
- ✅ CSV export with relevance tracking

### Upcoming Features (multi-source branch)
- 🔄 Google Scholar integration
- 🔄 PubMed integration
- 🔄 Unified search interface
- 🔄 Cross-source paper deduplication

## Setup

1. Install dependencies:
```bash
pip install crewai requests beautifulsoup4
```

2. Set up your preferred LLM provider:

### Ollama (Default)
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull the model
ollama pull llama3.1:8b
```

### Google Gemini
```bash
# Set your Gemini API key
export GEMINI_API_KEY=your_api_key_here
```

### Hugging Face
```bash
# Set your Hugging Face token
export HF_TOKEN=your_token_here
```

## Usage

```python
from research_pipeline import ResearchPipeline

# Initialize the pipeline
pipeline = ResearchPipeline(
    provider="gemini",  # or "ollama" or "huggingface"
    model_name="gemini-2.0-flash"
)

# Run the pipeline with your project description
result = pipeline.run(project_description="Your research topic description here")
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 