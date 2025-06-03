# API Delay Configuration Guide

The PaperPulse research pipeline now supports configurable API call delays to help you avoid rate limiting and be respectful to external services.

## Configuration Options

### Environment Variables

You can set the following environment variables to control API delays:

- `API_CALL_DELAY`: Seconds to wait between API calls (default: 60)
- `DOWNLOAD_WAIT_TIME`: Seconds to wait for Selenium downloads (default: 60)
- `LLM_CALL_DELAY`: Seconds to wait between LLM calls (default: 5)
- `LLM_RETRY_DELAY`: Seconds to wait before retrying after LLM rate limit (default: 30)
- `LLM_MAX_RETRIES`: Maximum number of retries for LLM calls (default: 3)

### Setting Environment Variables

#### Linux/Mac (Terminal):
```bash
export API_CALL_DELAY=90
export DOWNLOAD_WAIT_TIME=90
export LLM_CALL_DELAY=10
export LLM_RETRY_DELAY=60
python research_pipeline.py
```

#### Windows (Command Prompt):
```cmd
set API_CALL_DELAY=90
set DOWNLOAD_WAIT_TIME=90
set LLM_CALL_DELAY=10
set LLM_RETRY_DELAY=60
python research_pipeline.py
```

#### Windows (PowerShell):
```powershell
$env:API_CALL_DELAY="90"
$env:DOWNLOAD_WAIT_TIME="90"
$env:LLM_CALL_DELAY="10"
$env:LLM_RETRY_DELAY="60"
python research_pipeline.py
```

### Using a .env file

Create a `.env` file in your project directory:
```
API_CALL_DELAY=90
DOWNLOAD_WAIT_TIME=90
LLM_CALL_DELAY=10
LLM_RETRY_DELAY=60
```

Then load it in your Python script or use a package like `python-dotenv`.

## What These Delays Control

### API_CALL_DELAY (Default: 60 seconds)
This delay is used between:
- PubMed API calls when searching for papers
- Google Scholar scraping requests  
- arXiv API requests
- HTTP requests to PMC article pages
- After NCBI Entrez API errors
- Between processing each paper in download loops

### DOWNLOAD_WAIT_TIME (Default: 60 seconds)  
This delay is used for:
- Waiting for Selenium-driven PDF downloads to complete

### LLM Rate Limiting (New!)
The system now includes intelligent rate limiting specifically for LLM calls to handle quotas and avoid 429 errors:

**LLM_CALL_DELAY (Default: 5 seconds)**
- Small delay between individual LLM calls within processing loops
- Helps stay within rate limits during bulk processing

**LLM_RETRY_DELAY (Default: 30 seconds)**  
- Initial wait time before retrying after a rate limit error
- Uses exponential backoff for subsequent retries
- Automatically extracts suggested retry delays from API error responses

**LLM_MAX_RETRIES (Default: 3)**
- Maximum number of retry attempts for failed LLM calls
- After this limit, the operation will be marked as "FAILED_AFTER_RETRIES"

## Recommended Values

| Service/Use Case | Recommended API_CALL_DELAY | Notes |
|-----------------|---------------------------|-------|
| Light usage | 45-60 seconds | Default, respectful rate |
| Heavy usage | 90-120 seconds | Conservative, avoids rate limits |
| Debugging | 30 seconds | Faster for testing (use sparingly) |
| Production batch | 120+ seconds | Very conservative for large datasets |

### LLM Rate Limiting Recommendations

| LLM Provider/Tier | LLM_CALL_DELAY | LLM_RETRY_DELAY | LLM_MAX_RETRIES | Notes |
|------------------|----------------|-----------------|-----------------|-------|
| Gemini Free Tier | 10-15 seconds | 60 seconds | 5 | 15 requests/minute limit |
| Gemini Paid | 3-5 seconds | 30 seconds | 3 | Higher quota |
| Ollama Local | 1-2 seconds | 10 seconds | 2 | Local processing |
| OpenAI Free | 8-10 seconds | 45 seconds | 4 | Moderate limits |
| OpenAI Paid | 2-5 seconds | 20 seconds | 3 | Higher quota |

## Previous Configuration

Before this update, all delays were hardcoded to 50 seconds. The new default of 60 seconds is slightly more conservative while still being configurable.

## Tips

1. **Start Conservative**: Begin with higher delays (90+ seconds) and reduce if needed
2. **Monitor Logs**: Watch for rate limiting errors and adjust accordingly
3. **Respect APIs**: Different services have different rate limits - be courteous
4. **Batch Processing**: For large datasets, use higher delays and run overnight
5. **Test First**: Use smaller datasets to find optimal delay values

## Example Usage

For a conservative setup with 2-minute delays:
```bash
export API_CALL_DELAY=120
export DOWNLOAD_WAIT_TIME=90
export LLM_CALL_DELAY=10
export LLM_RETRY_DELAY=60
python research_pipeline.py
```

For faster development/testing (use sparingly):
```bash
export API_CALL_DELAY=30
export DOWNLOAD_WAIT_TIME=45  
export LLM_CALL_DELAY=5
export LLM_RETRY_DELAY=30
python research_pipeline.py
```

## Troubleshooting Rate Limit Issues

### Gemini Free Tier "Too Many Requests" Error

If you're seeing this error:
```
"You exceeded your current quota, please check your plan and billing details"
```

**Immediate Fix:**
```bash
export LLM_CALL_DELAY=15     # Wait 15 seconds between LLM calls
export LLM_RETRY_DELAY=90    # Wait 90 seconds before retrying
export LLM_MAX_RETRIES=5     # Try up to 5 times
python research_pipeline.py
```

**For Very Conservative Processing:**
```bash
export LLM_CALL_DELAY=20     # 20 seconds between calls
export LLM_RETRY_DELAY=120   # 2 minutes retry delay
export LLM_MAX_RETRIES=3     # 3 retry attempts
python research_pipeline.py
```

The system will now automatically:
- Pause between each LLM call
- Detect rate limit errors
- Wait the appropriate time before retrying
- Use exponential backoff for persistent errors
- Mark papers as "FAILED_AFTER_RETRIES" if all attempts fail 