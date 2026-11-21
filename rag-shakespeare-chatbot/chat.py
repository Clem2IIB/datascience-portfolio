import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import numpy as np
import json
import re

# paths and collection name are shared with fill_db.py, so both files always use the same database
from fill_db import BASE_DIR, CHROMA_PATH, MODEL_PATH, CATALOGUE_PATH, COLLECTION_NAME
# text cleaning is shared with split_text.py, so questions are cleaned like the titles and names of the catalogue
from split_text import normalize

LM_STUDIO_MODEL = "mistralai/ministral-3-3b" # default model to use for the chatbot but can be changed on the interface
# path to the prompt engineering used to build the system prompt with the retrieved context
PROMPT_PATH = BASE_DIR / "prompt_engineering.md"

# The LLM can run in two places: locally in LM Studio (free, needs a strong computer) or with the Claude API (needs an API key)
LM_STUDIO = "Local (LM Studio)"
CLAUDE_API = "Claude API (Anthropic)"
# reads the API key from the .env file next to this script (ANTHROPIC_API_KEY=...), only needed for the Claude API
load_dotenv(BASE_DIR / ".env")

AUTO = "Auto (detect the play)" # sidebar choices besides the list of works
ALL_WORKS = "All works"
MAX_WORKS_PER_NAME = 3 # a name found in more works than this ("KING", "MESSENGER") does not tell which play the question is about
MAX_CHUNKS_PER_WORK = 2 # when searching all works, keep at most this many chunks per work so a broad question gets several works
N_CANDIDATES = 50 # number of chunks each search (semantic and BM25) proposes before they are merged
RRF_K = 60 # constant of the Reciprocal Rank Fusion formula 1 / (k + rank), 60 is the usual value
FOLLOW_UP_WORDS = ["HE", "SHE", "HIM", "HER", "HIS", "THEY", "THEM", "THEIR", "IT", "NEXT", "THY"] # words of a question that continues the previous one

# Module level setup for the chatbot, including loading the local model and connecting to ChromaDB collection
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
claude_client = Anthropic() # takes the key ANTHROPIC_API_KEY loaded from the .env file
chromadb_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chromadb_client.get_or_create_collection(name=COLLECTION_NAME)
embedding_model = SentenceTransformer(str(MODEL_PATH))

# catalogue of works and character names written by fill_db.py
with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
    catalogue = json.load(f)
WORK_TITLES = [work["title"] for work in catalogue["works"]]


def detect_works(user_query):
    """
    Finds the works a question is about, from the work titles and character names it contains
    "Why does Iago hate Othello?" -> Othello. A name shared by a few plays ("Portia") gives all of them

    Args:
        user_query (str): question asked by the user

    Returns:
        list[str]: titles of the works found, empty if the question names no work or character
    """
    # spaces around the text so " IAGO " only matches the whole word, not a part of another word
    query = f" {normalize(user_query)} "
    found = []

    # short titles: "MACBETH", "KING LEAR", "TEMPEST", "SONNETS" (the singular counts too: "sonnet 18")
    for work in catalogue["works"]:
        for title_form in [work["short_title"], work["short_title"].rstrip("S")]:
            if f" {title_form} " in query:
                found.append(work["title"])
                # remove the title from the question, so "CAESAR" in "JULIUS CAESAR" is not counted again as a name
                query = query.replace(f" {title_form} ", " ")
                break

    # character names, longest first, so "LADY MACBETH" is found before "MACBETH"
    for name in sorted(catalogue["characters"], key=len, reverse=True):
        if f" {name} " in query:
            query = query.replace(f" {name} ", " ")
            works = catalogue["characters"][name]
            # ignore the names shared by too many plays
            if len(works) <= MAX_WORKS_PER_NAME:
                for title in works:
                    if title not in found:
                        found.append(title)

    return found


def choose_works(user_query, choice=AUTO, previous_works=None):
    """
    Decides in which works to search, from the sidebar choice and the question

    Args:
        user_query (str): question asked by the user
        choice (str): sidebar choice: AUTO, ALL_WORKS or one work title
        previous_works (list[str]): works used for the previous question of the conversation

    Returns:
        tuple: (works, reason): list of work titles (empty means all works) and a short explanation shown to the user
    """
    if choice == ALL_WORKS:
        return [], "all works (chosen in the sidebar)"
    if choice != AUTO:
        return [choice], "chosen in the sidebar"

    detected = detect_works(user_query)
    if detected:
        return detected, "detected in the question"

    # follow-up questions like "What does he do next?" name nobody, so we stay in the same works
    # a question counts as a follow-up when it uses a pronoun that refers to the previous answer
    is_follow_up = any(word in FOLLOW_UP_WORDS for word in normalize(user_query).split())
    if previous_works and is_follow_up:
        return previous_works, "kept from the previous question"

    return [], "all works (no play detected in the question)"


def tokenize(text):
    """Splits a text into lowercase words for the BM25 search ("Out, damned spot!" -> ["out", "damned", "spot"])"""
    return re.findall(r"[a-z]+", text.lower())


# all the chunks kept in memory: the BM25 index needs every text, and neighbours are found by their chunk_id
all_chunks = collection.get()
chunks_by_id = {}
for doc, meta in zip(all_chunks["documents"], all_chunks["metadatas"]):
    chunks_by_id[meta["chunk_id"]] = {"text": doc, "metadata": meta}
chunk_ids = list(chunks_by_id.keys())

# the BM25 index needs every chunk as a list of words, in the same order as chunk_ids
tokenized_chunks = []
for chunk_id in chunk_ids:
    tokenized_chunks.append(tokenize(chunks_by_id[chunk_id]["text"]))
bm25 = BM25Okapi(tokenized_chunks) # set up the BM25 index of all the chunks, used for the lexical search


def search_chunks(user_query, works=None, n_chunks=12, method="hybrid", max_per_work=MAX_CHUNKS_PER_WORK):
    """
    Retrieves the most relevant chunks for a given query with a hybrid search:
    a semantic search with the embeddings in ChromaDB (finds passages with the same meaning)
    and a lexical search with BM25 (finds passages with the same rare words, like the words of a quote),
    merged with Reciprocal Rank Fusion: a chunk ranked high in one or both searches comes first

    Args:
        user_query (str): question asked by the user
        works (list[str]): titles of the works to search in, all works if empty
        n_chunks (int): Number of chunks to retrieve. Defaults to 12
        method (str): "hybrid", or "semantic" / "lexical" to use one search alone (used in rag_evaluation.ipynb)
        max_per_work (int): when searching all works, maximum number of chunks per work (None for no limit)

    Returns:
        list[dict]: the chunks found, best first, each with its "text" and "metadata"
    """
    # 1. Semantic search: embed the question with the same local model SentenceTransformer used in fill_db.py
    query_embedding = embedding_model.encode(user_query).tolist()
    if works:
        # search only in the chosen works with a metadata filter
        results = collection.query(query_embeddings=[query_embedding], n_results=N_CANDIDATES, where={"work": {"$in": works}})
    else:
        results = collection.query(query_embeddings=[query_embedding], n_results=N_CANDIDATES)
    semantic_ids = [meta["chunk_id"] for meta in results["metadatas"][0]]

    # 2. Lexical search: BM25 scores every chunk, rare words of the question count more than common ones
    # BM25 is a ranking function that uses the frequency of a term in a document and the frequency of the term across all documents to rank documents.
    query_words = tokenize(user_query)
    if works:
        # inside the chosen works, their title and character names are in almost every chunk, so they do not help to rank
        # them: they are left out to keep the words that say what the question is about ("hate" in "Why does Iago hate Othello?")
        name_words = []
        for name, name_works in catalogue["characters"].items():
            if any(work in works for work in name_works):
                name_words += tokenize(name)
        for work in catalogue["works"]:
            if work["title"] in works:
                name_words += tokenize(work["short_title"])
        query_words = [word for word in query_words if word not in name_words]
        
    scores = bm25.get_scores(query_words) # gets a score for every chunk, higher means more relevant
    lexical_ids = []
    for position in np.argsort(scores)[::-1]: # positions of the chunks from the best score to the worst
        chunk_id = chunk_ids[position]
        if not works or chunks_by_id[chunk_id]["metadata"]["work"] in works:
            lexical_ids.append(chunk_id)
        if len(lexical_ids) == N_CANDIDATES:
            break

    # 3. Reciprocal Rank Fusion: each search gives 1 / (RRF_K + rank) to its chunks, then the scores are added
    # RRF is a simple way to merge two rankings, giving more weight to the chunks that are ranked high in one or both searches
    if method == "semantic":
        rankings = [semantic_ids] # brackets are needed to make a list of rankings, because the RRF loop 
    elif method == "lexical":
        rankings = [lexical_ids] 
    else:
        rankings = [semantic_ids, lexical_ids]

    rrf_scores = {}
    for ranking in rankings: # for each ranking, the best chunk gets 1/(RRF_K + 1), the second 1/(RRF_K + 2), etc.
        for rank, chunk_id in enumerate(ranking, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
    best_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True) # sort the chunk_ids by their RRF score, best first

    # 4. Keep the n_chunks best, with at most max_per_work chunks per work when searching all works
    chunks = []
    chunks_per_work = {} # number of chunks already kept for each work
    for chunk_id in best_ids:
        chunk = chunks_by_id[chunk_id]
        count = chunks_per_work.get(chunk["metadata"]["work"], 0)
        if not works and max_per_work and count >= max_per_work:
            continue
        chunks_per_work[chunk["metadata"]["work"]] = count + 1
        chunks.append(chunk)
        if len(chunks) == n_chunks:
            break

    return chunks


def add_neighbours(chunks):
    """
    Adds to each chunk the chunk just before and just after it, when they are in the same scene (or poem)
    A chunk is often one speech: the speeches around it give the LLM the rest of the dialogue

    Args:
        chunks (list[dict]): chunks returned by search_chunks()

    Returns:
        list[dict]: one passage per retrieved chunk, with its neighbours joined around it
    """
    passages = []
    used_ids = set(chunk["metadata"]["chunk_id"] for chunk in chunks) # avoids repeating a chunk already in another passage
    for chunk in chunks:
        meta = chunk["metadata"]
        texts = [chunk["text"]]

        # neighbouring chunks have neighbouring IDs, because fill_db.py numbers the chunks in the order of the text
        previous_id = meta["chunk_id"] - 1
        next_id = meta["chunk_id"] + 1

        # the chunk just before goes in front of the text
        if is_good_neighbour(previous_id, meta, used_ids):
            texts.insert(0, chunks_by_id[previous_id]["text"])
            used_ids.add(previous_id)

        # the chunk just after goes behind it
        if is_good_neighbour(next_id, meta, used_ids):
            texts.append(chunks_by_id[next_id]["text"])
            used_ids.add(next_id)

        passages.append({"text": "\n\n".join(texts), "metadata": meta})

    return passages


def is_good_neighbour(neighbour_id, meta, used_ids):
    """
    Tells if a neighbouring chunk can be added to a passage: it exists, it is not already used,
    and it comes from the same work and the same scene (or poem), so the label of the passage stays true

    Args:
        neighbour_id (int): chunk_id of the neighbour
        meta (dict): metadata of the retrieved chunk
        used_ids (set): chunk_ids already in a passage

    Returns:
        bool: True if the neighbour can be added
    """
    if neighbour_id not in chunks_by_id or neighbour_id in used_ids:
        return False

    neighbour_meta = chunks_by_id[neighbour_id]["metadata"]
    same_work = neighbour_meta["work"] == meta["work"]
    same_section = neighbour_meta["section"] == meta["section"]
    return same_work and same_section


def get_context(user_query, works=None, n_chunks=12, neighbours=True):
    """
    Retrieves the most relevant passages and prepares them into a context string, each passage tagged with its work, Act and Scene

    Args:
        user_query (str): question asked by the user
        works (list[str]): titles of the works to search in, all works if empty
        n_chunks (int): Number of chunks to retrieve. Defaults to 12
        neighbours (bool): add the chunks around each retrieved chunk

    Returns:
        str: the passages wrapped in <passage> XML tags
    """
    passages = search_chunks(user_query, works, n_chunks)
    if neighbours:
        passages = add_neighbours(passages)

    # XML tags show the LLM clearly where each passage starts and ends and where it comes from
    context = ""
    for passage in passages:
        meta = passage["metadata"]
        source = f"{meta['work']}, {meta['section']}" if meta["section"] else meta["work"]
        context += f'<passage source="{source}">\n{passage["text"]}\n</passage>\n\n'

    return context


def load_prompt_template():
    """Loads the system prompt template from prompt_engineering.md."""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_system_prompt(context):
    """
    Joins the full system prompt by combining the retrieved context with the prompt template in prompt_engineering.md.
    Prompt template contains instructions for the LLM on how to use the context and answer the user's question
    Retrieved context are the formatted chunks at the end of the prompt to provide the necessary information for the LLM to generate an informed response only based on the provided data

    Args:
        context (str): formatted passages returned by get_context().

    Returns:
        str: complete system prompt to send to the LLM.
    """
    prompt_template = load_prompt_template()
    system_prompt = prompt_template + "\n\n<passages>\n" + context + "</passages>"

    return system_prompt


def call_llm(messages, provider=LM_STUDIO, model=LM_STUDIO_MODEL):
    """
    Sends the messages to the LLM and returns its answer, either locally with LM Studio or with the Claude API
    The two services take the same messages but in a slightly different format, so each has its own block

    Args:
        messages (list[dict]): the system prompt first, then the conversation, as [{"role": ..., "content": ...}]
        provider (str): LM_STUDIO or CLAUDE_API
        model (str): model identifier of the chosen service

    Returns:
        str: the answer of the LLM
    """
    if provider == LM_STUDIO:
        # LM Studio takes the system prompt as the first message, like the original version of the chatbot
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content

    # The Claude API takes the system prompt in its own "system" parameter, and only user and assistant messages
    system_prompt = messages[0]["content"]
    conversation = messages[1:]

    response = claude_client.messages.create(
        model=model,
        max_tokens=8000, # maximum length of the answer, set high so a long answer is not cut
        system=system_prompt,
        messages=conversation
    )

    # the model can decline a question for safety reasons, then there is no answer to show
    if response.stop_reason == "refusal":
        return "The model declined to answer this question."

    # the answer is a list of blocks (text, but also thinking blocks), only the text blocks are the answer
    answer = ""
    for block in response.content:
        if block.type == "text":
            answer += block.text
    return answer


def get_response(user_query, history=None, model=LM_STUDIO_MODEL, n_chunks=12, works=None, neighbours=True,
                 provider=LM_STUDIO):
    """
    Retrieves relevant context from ChromaDB, builds the system prompt, appends conversation history, and calls the LLM (LM Studio or Claude API)
    Gives a response to the user's query using the RAG pipeline

    Args:
        user_query (str): question asked by the user
        history (list[dict]): previous conversation messages in the format [{"role": ..., "content": ...} to provide conversation context to the LLM
        model (str): model identifier to use (LM Studio or Claude)
        n_chunks (int): number of chunks to retrieve
        works (list[str]): titles of the works to search in, all works if empty
        neighbours (bool): add the chunks around each retrieved chunk
        provider (str): LM_STUDIO or CLAUDE_API

    Returns:
        tuple: (response, context): the response of the LLM as a string, and the passages it received
    """

    # retrieve the context, the system prompt and message history
    context = get_context(user_query, works, n_chunks, neighbours)
    system_prompt = build_system_prompt(context)
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        # keeps only 4 last messages to avoid too much context that slows the response time
        # and only their role and content, the interface stores other information in the messages
        for message in history[1:][-4:]:
            messages.append({"role": message["role"], "content": message["content"]})

    # update the user query with the conversation history to provide more context
    messages.append({"role": "user", "content": user_query})

    # call the LLM (LM Studio or Claude API) and return the response
    response = call_llm(messages, provider, model)
    return response, context


if __name__ == "__main__":
    user_query = input("What do you want to know about Shakespeare's works?\n\n")
    works, reason = choose_works(user_query)
    print(f"Searching in: {', '.join(works) if works else 'all works'} ({reason})\n")
    response, context = get_response(user_query, works=works)
    print(response)
