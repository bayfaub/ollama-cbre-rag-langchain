# RAG Gen AI Application aimed at answering Commercial Real Estate questions

## Project Goal
1. Gain a basic knowledge of LangChain
2. Leverage an open-source LLM to build a small local RAG Application
3. Learn as much as possible
4. Expand the project into a more complex agent(work in progess)

## Local Project Setup
 1. Run ```pip install requirements.txt``` make sure you're running python 3.10 in a virtual environment
 2. Download [Ollama](https://ollama.com/download)
 3. Install the Gemma2 9B model using `Ollama pull gemma2`
 4. Install the nomic-embed-text model using `Ollama pull nomic-embed-text`
 5. Run the indexer to populate your local chromadb vectorstore using `python indexer.py` by default this will use the "large" document directory if you want to change it use the small directory simply edit the `documents_directory` variable in `indexer.py` to be ```documents_directory = "./documents/small"```

 If you have any questions please email `bay@faubionstreet.com`
