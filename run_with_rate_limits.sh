#!/bin/bash

# PaperPulse Rate Limiting Configuration Script
# This script sets conservative rate limits to avoid API quota issues

echo "Setting up PaperPulse with conservative rate limiting..."

# External API delays (PubMed, Google Scholar, arXiv)
export API_CALL_DELAY=60
export DOWNLOAD_WAIT_TIME=60

# LLM Rate Limiting (Conservative settings for Gemini Free Tier)
export LLM_CALL_DELAY=15        # 15 seconds between LLM calls
export LLM_RETRY_DELAY=90       # 90 seconds before retrying after rate limit
export LLM_MAX_RETRIES=5        # Try up to 5 times

echo "Rate limiting configured:"
echo "  API_CALL_DELAY: $API_CALL_DELAY seconds"
echo "  DOWNLOAD_WAIT_TIME: $DOWNLOAD_WAIT_TIME seconds" 
echo "  LLM_CALL_DELAY: $LLM_CALL_DELAY seconds"
echo "  LLM_RETRY_DELAY: $LLM_RETRY_DELAY seconds"
echo "  LLM_MAX_RETRIES: $LLM_MAX_RETRIES attempts"
echo ""
echo "Starting PaperPulse..."

python research_pipeline.py 