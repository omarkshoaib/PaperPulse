from crewai import Agent, Task, Crew, Process
from research_tools import ArxivSearchTool, CSVWriterTool
from typing import List, Dict, Any
from crewai import LLM
import os
import requests
from typing import Optional
import json
import time # Import time module

class ResearchPipeline:
    def __init__(self, 
                 provider: str = "gemini",  # Options: "ollama", "gemini", "huggingface"
                 model_name: str = "gemini-2.0-flash",
                 base_url: str = "http://localhost:11434"):
        """
        Initialize the research pipeline with the specified LLM provider.
        
        Args:
            provider: The LLM provider to use ("ollama", "gemini", or "huggingface")
            model_name: The specific model to use
            base_url: Base URL for the provider (only used for Ollama)
        """
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        
        # Configure LLM based on provider
        if provider == "ollama":
            # Verify Ollama is running and model is available
            self._verify_ollama_setup(base_url, model_name)
            self.local_llm = LLM(
                model=f"ollama/{model_name}",
                base_url=base_url,
                temperature=0.7,
                timeout=300
            )
        elif provider == "gemini":
            # Configure Gemini LLM
            self.local_llm = LLM(
                model="gemini/gemini-2.0-flash",
                temperature=0.7,
                timeout=300
            )
        elif provider == "huggingface":
            # Configure Hugging Face LLM
            self.local_llm = LLM(
                model="huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct",
                temperature=0.7,
                timeout=300
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'ollama', 'gemini', or 'huggingface'")
        
        # Initialize tools
        self.arxiv_tool = ArxivSearchTool(max_results=10) # Temporarily reduced for testing
        self.csv_tool = CSVWriterTool()
        
        # Create agents
        self.query_agent = Agent(
            role="Research Query Generator",
            goal="Generate effective search queries for research papers",
            backstory="You are an expert at formulating precise search queries for academic research",
            verbose=True,
            allow_delegation=False,
            tools=[self.arxiv_tool],
            llm=self.local_llm
        )
        
        self.summarizer_agent = Agent(
            role="Research Summarizer",
            goal="Create dense, informative summaries of research papers",
            backstory="You are an expert at distilling complex research into clear, concise summaries",
            verbose=True,
            allow_delegation=False,
            llm=self.local_llm
        )
        
        self.validator_agent = self.create_validator_agent()

    def _verify_ollama_setup(self, base_url: str, model_name: str) -> None:
        """Verify that Ollama is running and the model is available."""
        try:
            # Check if Ollama server is running
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Ollama server is not running. Please start it with 'ollama serve'")
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [model.get('name') for model in models]
            if model_name not in model_names:
                print(f"Model {model_name} not found. Available models: {model_names}")
                print(f"Please pull the model with: ollama pull {model_name}")
                raise ValueError(f"Model {model_name} not found in Ollama")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Could not connect to Ollama server. Please ensure it's running with 'ollama serve'")

    def create_validator_agent(self):
        return Agent(
            role='Research Validator',
            goal='Critically evaluate the relevance of a research paper to a given topic',
            backstory="""You are a meticulous research validation expert. Your sole focus is to determine
            if a given research paper (title, summary, keywords) is highly relevant to a specific research topic.
            You do not perform any other actions other than relevance validation.""",
            verbose=True,
            allow_delegation=False,
            llm=self.local_llm
        )

    def create_tasks(self):
        query_task_description = """Generate effective arXiv search queries to find research papers covering the following key areas relevant to a larger project on automated video analysis for form classification:
example:        
1.  **MediaPipe for Pose Estimation & Data Preparation**: Papers discussing the use of MediaPipe for extracting human pose data from videos, and any subsequent data preparation or feature engineering steps (e.g., joint angles, velocities) based on MediaPipe output.
2.  **Graph Convolutional Networks (GCNs) for Pose-Based Analysis**: Research on GCN architectures (Spatial, Temporal, Spatio-Temporal) applied to human pose data for tasks like action recognition, form classification, or movement analysis. Include papers that discuss converting pose data into graph representations for GCN input.
3.  **LSTM Models for Pose Sequence Modeling**: Papers that utilize LSTM (Long Short-Term Memory) networks for modeling and analyzing temporal sequences of human pose data.

Your goal is to find papers that are strong in at least one of these specific technical areas.
Prioritize recent advancements and practical implementations.

Use the ArxivSearchTool to search for papers and return a list of papers in this format:
[
    {
        'title': 'paper title',
        'link': 'paper url',
        'abstract': 'paper abstract'
    },
    ...
]"""

        query_task = Task(
            description=query_task_description,
            agent=self.query_agent,
            expected_output="A list of research papers with their titles, links, and abstracts."
        )

        summarize_task = Task(
            description="""Summarize the research papers using the Chain of Density technique.
            Input format from previous task (a list of paper dictionaries):
            [
                {
                    'title': 'paper title',
                    'link': 'paper url',
                    'abstract': 'paper abstract'
                },
                ...
            ]
            
            For EACH paper in the provided list, create a dense summary of its abstract and extract relevant keywords.
            Your output MUST be a valid JSON list of objects. EACH object in the list MUST strictly follow this format:
            {
                "title": "ACTUAL paper title from input (ensure valid JSON string escaping for quotes or special characters)",
                "link": "ACTUAL paper link from input (ensure valid JSON string escaping)",
                "densified_abstract": "DETAILED and DENSE summary of the abstract using Chain of Density (ensure valid JSON string escaping for all special characters, newlines should be \n, quotes should be \")",
                "keywords": ["keyword1", "keyword2", "keyword3", ...] 
            }
            Ensure ALL strings are properly JSON escaped. Ensure commas are correctly placed: between elements in lists, and between key-value pairs in objects. DO NOT use trailing commas.
            Return ONLY the JSON list of these structured paper summaries. Do NOT include any other text, explanation, or markdown formatting like ```json before or after the list.
            Your entire response must be ONLY the JSON list itself, starting with '[' and ending with ']'.""",
            agent=self.summarizer_agent,
            expected_output="A list of papers, each with its original title, link, a new densified_abstract, and extracted keywords.",
            context=[query_task]  # Pass the query agent's output (papers) to the summarizer
        )
        
        # Removed validate_task from here, it will be handled in the run method
        return [query_task, summarize_task]

    def run(self, project_description: str):
        # Create initial tasks (query and summarize)
        query_task, summarize_task = self.create_tasks()

        # Run query and summarization crew
        initial_crew = Crew(
            agents=[self.query_agent, self.summarizer_agent],
            tasks=[query_task, summarize_task],
            verbose=True,
            process=Process.sequential
        )
        crew_output_obj = initial_crew.kickoff()

        print("\n--- Summarization Complete. Starting Validation and CSV Writing ---")
        
        summarized_papers = []
        raw_data = None

        if crew_output_obj:
            # The .raw attribute of CrewOutput should contain the raw string output of the last task
            raw_data = crew_output_obj.raw 
            if raw_data:
                try:
                    # Extract JSON part from the raw_data
                    json_block_start = raw_data.find("```json")
                    if json_block_start != -1:
                        # Found ```json, now find the end of this block
                        json_content_start = json_block_start + 7 # Skip ```json
                        json_block_end = raw_data.find("```", json_content_start)
                        if json_block_end != -1:
                            processed_data = raw_data[json_content_start:json_block_end].strip()
                        else:
                            # Fallback if ```json is present but no closing ```, might be risky
                            print("Warning: Found '```json' but no closing '```'. Attempting to parse from '```json' onwards.")
                            processed_data = raw_data[json_content_start:].strip()
                    else:
                        # Fallback if no ```json, try to find first '[' or '{' assuming raw output is just JSON
                        first_brace = raw_data.find('[')
                        first_curly = raw_data.find('{')

                        if first_brace != -1 and (first_curly == -1 or first_brace < first_curly) :
                            processed_data = raw_data[first_brace:]
                        elif first_curly != -1 and (first_brace == -1 or first_curly < first_brace) :
                            processed_data = raw_data[first_curly:]
                        else: # No clear JSON start found
                            processed_data = raw_data.strip() # Try as is, might fail

                    summarized_papers = json.loads(processed_data)
                    if not isinstance(summarized_papers, list):
                        print(f"Warning: Parsed summarizer output is not a list. Got: {type(summarized_papers)}")
                        print(f"Parsed data: {summarized_papers}")
                        summarized_papers = []
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from summarizer: {e}")
                    print(f"Raw summarizer output was: {raw_data}")
                    summarized_papers = [] 
                except Exception as e:
                    print(f"An unexpected error occurred while processing summarizer output: {e}")
                    print(f"Raw summarizer output was: {raw_data}")
                    summarized_papers = []
            else:
                print("Warning: Crew output's .raw attribute is empty after summarization.")
        else:
            print("Warning: Crew kickoff for summarization returned no output object.")

        if not summarized_papers:
             print("No summarized papers to process. Check warnings above.")

        saved_papers_count = 0
        final_report_parts = []

        for paper in summarized_papers:
            if not isinstance(paper, dict) or not all(k in paper for k in ['title', 'link', 'densified_abstract', 'keywords']):
                print(f"Skipping invalid paper data: {paper}")
                continue

            print(f"\nValidating paper: {paper.get('title', 'N/A')}")
            
            # Use the project_description passed to the run method for validation topic
            validation_task_description = f"""Critically evaluate if the following research paper is relevant to the topic: 
            '{project_description}'.
            
            Paper Details:
            Title: {paper.get('title')}
            Summary: {paper.get('densified_abstract')}
            Keywords: {paper.get('keywords')}
            
            Your answer MUST be exactly 'RELEVANT' or 'NOT RELEVANT'. Do not provide any other explanation, text, or punctuation.
            Your entire response must be ONLY the word RELEVANT or ONLY the word NOT RELEVANT."""
            
            validation_task = Task(
                description=validation_task_description,
                agent=self.validator_agent,
                expected_output="A single word: either 'RELEVANT' or 'NOT RELEVANT'." # Stricter expected output
            )
            
            # Crew for a single validation task
            validation_crew = Crew(
                agents=[self.validator_agent],
                tasks=[validation_task],
                verbose=False # Keep validation per paper less verbose in main log
            )
            
            validation_crew_kickoff_result = validation_crew.kickoff()
            actual_validator_response_str = "" 

            # Try to get the raw string output from the validation crew
            if hasattr(validation_crew_kickoff_result, 'raw') and validation_crew_kickoff_result.raw is not None:
                actual_validator_response_str = str(validation_crew_kickoff_result.raw).strip()
            elif isinstance(validation_crew_kickoff_result, str): # If kickoff somehow returns a plain string
                actual_validator_response_str = validation_crew_kickoff_result.strip()
            else:
                # Fallback if .raw is not useful, try converting the whole object to string (might contain the answer)
                print(f"Warning: validation_crew_kickoff_result.raw was not directly usable. Type: {type(validation_crew_kickoff_result)}. Trying str().")
                actual_validator_response_str = str(validation_crew_kickoff_result).strip()

            print(f"Validation result for '{paper.get('title')}': {actual_validator_response_str}")

            # Add relevance status to paper data
            paper['relevance'] = actual_validator_response_str.strip().upper()
            
            try:
                # Ensure keywords are a string for CSVWriterTool
                if isinstance(paper.get('keywords'), list):
                    paper['keywords'] = ', '.join(paper['keywords'])
                
                self.csv_tool._run(paper_data=paper) # Call CSV tool directly
                saved_papers_count += 1
                final_report_parts.append(f"Paper '{paper.get('title')}' saved ({paper['relevance']}).")
                print(f"Saved paper to CSV: {paper.get('title')} ({paper['relevance']})")
            except Exception as e:
                print(f"Error saving paper '{paper.get('title')}' to CSV: {e}")
            
            time.sleep(60)

        final_summary = f"\n--- Validation and CSV Writing Complete ---\
Saved {saved_papers_count} papers to CSV.\n"
        final_summary += "\nDetails:\n" + "\n".join(final_report_parts)
        
        print(final_summary)
        return final_summary

if __name__ == "__main__":
    try:
        # 1. Using Ollama
        # pipeline = ResearchPipeline(
        #     provider="ollama",
        #     model_name="llama3.1:8b",
        #     base_url="http://localhost:11434"
        # )
        
        # 2. Using Gemini (default)
        pipeline = ResearchPipeline(
            provider="gemini",
            model_name="gemini-2.0-flash"
        )
        
        # 3. Using Hugging Face
        # pipeline = ResearchPipeline(
        #     provider="huggingface",
        #     model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
        # )
        
        project_description = """This project introduces GCN-DAM, a novel architecture combining Graph Convolutional Networks (GCNs) with Large Language Models (LLMs) to enable interpretable human action evaluation. Traditional skeleton-based models like ST-GCN and 2s-AGCN achieve high accuracy but lack interpretability and actionable feedback. Inspired by Describe Anything Model (DAM) and MotionGPT, GCN-DAM uses focal prompts and gated cross-attention (GCA) to fuse skeleton data with RGB visual context. The architecture includes a dual-input focal prompt, graph-vision fusion module, hierarchical temporal encoder, and an LLM-based evaluator that generates multi-granular feedback. Through a motion-to-text projection, the system translates structured skeleton features into natural language using a structured prompt template. A semi-supervised data pipeline called Motion-SDP is used to overcome dataset limitations by expanding labeled datasets with self-supervised learning. Techniques like temporal warping, joint masking, and Gaussian noise improve model robustness and generalizability. Evaluation is performed using Motion-Bench, a reference-free benchmark inspired by DLC-Bench, focusing on attribute-based evaluation, temporal localization, and feedback utility. The model supports real-time deployment via knowledge distillation, quantization, and temporal sampling. Applications span sports, rehabilitation, safety monitoring, and performance arts, making GCN-DAM a promising direction for interpretable, multimodal, and adaptive action assessment systems."""
        result = pipeline.run(project_description)
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting steps:")
        if pipeline.provider == "ollama":
            print("1. Ensure Ollama is running with 'ollama serve'")
            print("2. Verify the model is installed with 'ollama list'")
            print("3. If needed, pull the model with 'ollama pull llama3.1:8b'")
        elif pipeline.provider == "gemini":
            print("1. Ensure GEMINI_API_KEY is set in your environment")
            print("2. Verify your API key is valid")
        elif pipeline.provider == "huggingface":
            print("1. Ensure HF_TOKEN is set in your environment")
            print("2. Verify your token has access to the model") 