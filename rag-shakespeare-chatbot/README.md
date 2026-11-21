# Shakespeare Chatbot (RAG)

A local chatbot that answers questions about Shakespeare's plays and poems, using only passages retrieved from the text.

This RAG project started as an academic group project on the Othello play only. I have extended the project to include all the complete works from Shakespeare thanks to the [Project Gutenberg](https://www.gutenberg.org/ebooks/100). It includes now 38 plays, the Sonnets and 5 other poems. 

## Members of the original Othello project
- Jules Charrier
- Louise Ignesta
- Ketan Prunet
- Hippolyte Maillot
- Clément de Barbeyrac

**Original project**: The original Othello chatbot was an academic group project. We designed the project and the structure together, I was working mostly on the code. For this portfolio version I worked alone to extend it to the full collection of Shakespeare's work, while keeping the chatbot's answers precise and accurate.

**Extension (solo)**:

- The complete works file is split automatically into 44 works, using its table of contents and fuzzy matching of titles.
- Each chunk has act and scene metadata (or the poem number for the Sonnets and poems), and long speeches are cut to fit the embedding model.
- The play is detected from the titles and character names in the question.
- Retrieval is a hybrid search (embeddings + BM25), and the chunks around each result are added.
- The model can run locally in LM Studio or through the Claude API.
- An evaluation notebook measures each step on 40 test questions. The final version finds the right passage for 30 of them with 12 chunks.

**A note on AI use:** 
- The original Othello chatbot was built by hand by the team. 
- For the extension to the complete works, the structure, decisions and code are my own work. 
- I used AI (Claude) as an assistant to tidy and reorganize the code's formatting and comments, or to implement few specific functions
- The evaluation notebook has been built step by step with the help of AI (Claude), to test and present the performances of the RAG

## Requirements

### 1. Choose where the LLM runs (one of the two, or both)

**Option A: locally with [LM Studio](https://lmstudio.ai/)**, free but needs a strong computer.
Run locally the following models on LM Studio, with the local server started on port 1234
   - Ministral 3B `mistralai/ministral-3-3b`
   - Gemma 2B `google/gemma-4-e2b`

**Option B: with the Claude API**, needs an Anthropic API key (paid per question).
Create a file named `.env` in the project folder with this line (the file is git-ignored, so the key stays on your computer):
```
ANTHROPIC_API_KEY=your-key-here
```
Model used: Claude Haiku 5.5, the cheapest Claude model.

### 2. Clone the repository from Github

### 3. Create a virtual environment
```bash
python -m venv env # create the environment

source env/bin/activate # activate it for Mac and Linux
env\Scripts\activate.bat # activate it for Windows
```

### 4. Install packages
```bash
pip install -r requirements.txt
```

### 5. Build the vector database in `fill_db.py` file
```bash
python fill_db.py
```
It downloads the embedding model once into `models/`, then embeds about 12,500 chunks (about 4 minutes on a laptop).

### 6. Launch the streamlit app to access the interface
```bash
streamlit run app.py
```
### 7. Navigate the interface
- A homepage to present chatbot and it purpose
- A chatpage to interact with the model containing
  - A text input for the user to ask questions
  - A "New Chat" button to clear the current discussion history and create a new chat
  - A choice of where the model runs (LM Studio or Claude API), then a drop down menu to select the model
  - A drop down menu to choose where to search: detect the play from the question, all works, or one work
  - A slider to select the number of chunks retrieved and used by the LLM to build its answer
  - A checkbox to add the passages around each retrieved chunk
  - Under each answer, the works searched and the passages used


> *it can take some time to load the chat input interface based on the performances of the computer*


## How it works concretely

- The chatbot uses a Retrieval-Augmented Generation (RAG) pipeline.
- `split_text.py` splits the complete works into 44 works using the table of contents of the file, then splits each work into dynamic chunks (takes the full speech without cutting in the middle, long speeches are cut between two lines of verse).
- `fill_db.py` gives each chunk its work, Act and Scene (or poem number) as metadata, and stores the chunks in a local ChromaDB vector database.
- When the user asks a question, `chat.py` looks for a play title or a character name in it to search in the right play. It retrieves the most relevant chunks with a hybrid search (semantic search with embeddings + BM25 lexical search, merged with Reciprocal Rank Fusion) and adds the chunks around them.
- The passages are passed as context to the LLM, local in LM Studio or through the Claude API, with the instructions in `prompt_engineering.md`.
- The model then answers questions based on those retrieved passages, with source references to the work, Act and Scene.

## Evaluation

`rag_evaluation.ipynb` explores the chunks and measures the retrieval on 40 test questions (`data/eval_questions.csv`): how often the passage that answers the question is sent to the LLM, for each step of the pipeline.
