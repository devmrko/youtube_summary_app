## YouTube Summarizer

This is a Python script that summarizes YouTube videos. It uses the YouTube API to get the transcript of the video and then uses a language model to summarize the transcript.
1. First example, it's using DeepL for translation and OpenAI/Ollama for the language model.
2. Second example, it's using OCI Translation and OCI LLM for the language model.

### Requirements

- Python 3.x    
- DeepL API key in .env file
- OpenAI API key in .env file

### configuration OCI credentials
- in .env file, edit OCI_CONFIG_PATH, OCI_KEY_PATH for your own

### Usage

- install dependencies
```
pip install -r requirements.txt
```

- apply python environment
```
# Ensure you're using the desired Python version via pyenv
pyenv local 3.10.7  

# Create a venv inside your project
python -m venv venv

.\.venv\Scripts\Activate.ps1
```

- run the script with a different video
```
python .\example_deepl_ollama_llm_test.py --video_url="https://www.youtube.com/watch?v=YULLBQKEYmM

python .\example_oci_llm_test.py --video_url="https://www.youtube.com/watch?v=YULLBQKEYmM"
```
