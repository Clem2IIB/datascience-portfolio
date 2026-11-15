from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
from string import capwords
import json
import re

# text -> chunks is done in split_text.py, this file builds the metadata and fills the database
from split_text import load_text, split_into_works, chunk_text, normalize, ACT_PATTERN, SCENE_PATTERN, SPEAKER_PATTERN

# Paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"
CATALOGUE_PATH = BASE_DIR / "data" / "catalogue.json" # list of works and character names, used by chat.py to detect the play
MODEL_PATH = BASE_DIR / "models"

# Embedding model and database
MODEL_NAME = "all-MiniLM-L6-v2" # small local embedding model, downloaded once into MODEL_PATH
COLLECTION_NAME = "shakespeare"
SOURCE = "Project Gutenberg #100"
BATCH_SIZE = 1000 # ChromaDB refuses too many documents in one upsert, so they are added in batches
SEARCH_PRECISION = 1000 # "ef_search" of ChromaDB: with its default (100) the search is approximate and changes between runs

# Threshold used to find the character names
MAX_WORKS_MENTIONING_NAME = 10 # a character name written in more works than this is an ordinary word ("ROMAN"), not a character


def to_roman(number):
    """
    Writes an act or scene number in Roman numerals, so all plays use the same style
    Some plays write "SCENE 2." instead of "SCENE II.": "2" -> "II", Roman numerals are kept as they are

    Args:
        number (str): the number found in the header

    Returns:
        str: the number in Roman numerals
    """
    roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                      "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]

    if number.isdigit() and 1 <= int(number) <= len(roman_numerals):
        return roman_numerals[int(number) - 1]
    return number


def readable_title(raw_title):
    """
    Writes a title in the usual way: "OTHELLO, THE MOOR OF VENICE" -> "Othello, the Moor of Venice"

    Args:
        raw_title (str): title in capitals, as written in the text

    Returns:
        str: the title with capitals only on the main words
    """
    title = capwords(raw_title.lower())
    for small_word in ["Of", "The", "And", "A", "Or", "For"]:
        title = title.replace(f" {small_word} ", f" {small_word.lower()} ")
    return title


def short_title(raw_title):
    """
    Gives the short name people use for a work, to recognise it in a question
    "THE TRAGEDY OF MACBETH" -> "MACBETH", "OTHELLO, THE MOOR OF VENICE" -> "OTHELLO"

    Args:
        raw_title (str): title in capitals, as written in the text

    Returns:
        str: the normalized short title
    """
    # words removed from the start of a title
    title_prefixes = ["THE TRAGEDY OF ", "THE LIFE OF ", "THE HISTORY OF ", "THE ", "A "]

    # keep only the part before the first comma, colon or semicolon
    main_part = re.split(r"[,:;]", raw_title)[0]
    title = normalize(main_part)
    for prefix in title_prefixes:
        if title.startswith(prefix):
            title = title[len(prefix):] # len(prefix) is the number of characters to remove from the start of the title
    return title


def build_catalogue(works):
    """
    Builds the catalogue used by chat.py to recognise the works a question is about
    It lists every work with its short title, and the character names of each play with the plays where they speak
    Two kinds of character names would be found in questions about other plays, so they are skipped:
    - names made only of common words ("LADY", "FIRST CITIZEN", "SAY")
    - names written in the text of many works, used as an ordinary word ("ROMAN") or a reference ("CAESAR")

    Args:
        works (list[dict]): the works returned by split_into_works()

    Returns:
        dict: {"works": [...], "characters": {name: [titles]}}
    """
    catalogue = {"works": [], "characters": {}}

    # 1. Every word written in lowercase somewhere in the works ("lady", "say", "nurse"), to find names that are common words
    all_text = " ".join(work["text"] for work in works)
    lowercase_words = set(re.findall(r"[a-z]+", all_text))

    # 2. The text of each work in capitals, to count in how many works a name is written 
    # "ROMAN" is written in many works, so it is not a character name, but "LADY MACBETH" is only in one play
    normalized_texts = []
    for work in works: 
        normalized_texts.append(f" {normalize(work['text'])} ") # normalize adds spaces at the start and end so "ROMAN" is not found in "ROMANCE" for example

    for work in works:
        title = readable_title(work["title"])
        # same rule as in build_metadata: a work is a play if it has at least one act header
        work_type = "play" if ACT_PATTERN.search(work["text"]) else "poetry"

        # 3. The work itself, with its short title
        catalogue["works"].append({"title": title, "short_title": short_title(work["title"]), "type": work_type})

        if work_type != "play":
            continue

        # 4. Every speaker name of the play (a set keeps each name once)
        speaker_names = set()
        for name in SPEAKER_PATTERN.findall(work["text"]):
            name = normalize(name)
            # "A ROMAN" is a role, the article is removed so it is checked as "ROMAN"
            name = re.sub(r"^(A|AN|THE) ", "", name)
            speaker_names.add(name)

        # 5. Keep the names that are real characters, with the list of plays where they speak
        for name in speaker_names:
            is_common_words = True
            for word in name.split():
                if word.lower() not in lowercase_words:
                    is_common_words = False

            works_mentioning = 0
            for text in normalized_texts:
                if f" {name} " in text:
                    works_mentioning += 1

            if not is_common_words and works_mentioning <= MAX_WORKS_MENTIONING_NAME:
                catalogue["characters"].setdefault(name, []).append(title)

    return catalogue


def build_metadata(works):
    """
    Builds the texts, IDs, and metadata lists for ChromaDB injection
    Detects when Act and Scene headers are present in each chunk using regex so every chunk knows its position in the play
    Poems have no acts, so a number alone on a line (Sonnet 18) is used instead
    Also builds the catalogue of works and names, used by chat.py to recognise the play a question is about

    Args:
        works (list[dict]): the works returned by split_into_works()

    Returns:
        tuple: (texts, ids, metadatas, catalogue): three parallel lists ready for upsert into ChromaDB, and the catalogue of works and names
    """
    texts = [] # list of chunk texts
    metadatas = [] # list of dicts with keys: source, chunk_id, work, type, act, scene, section
    ids = [] # list of unique IDs for each chunk

    # number alone on a line, used to number poems
    number_line_pattern = re.compile(r"^\s*([IVXL]+|\d+)\.?\s*$") # matches a line that contains only a number (Roman or Arabic) with optional whitespace and an optional period

    chunk_id = 0 # counts the chunks across all the works, so neighbouring chunks have neighbouring IDs

    for work in works:
        title = readable_title(work["title"])

        # a work is a play if it has at least one act header, otherwise it is poetry
        work_type = "play" if ACT_PATTERN.search(work["text"]) else "poetry"

        current_act = "" # temporary variable to track the current act
        current_scene = "" # temporary variable to track the current scene
        current_number = "" # temporary variable to track the current poem number

        # iterate through the chunks and build the metadata by detecting Act and Scene headers with regex to then append the lists
        for chunk in chunk_text(work["text"]):
            if work_type == "play":
                act_match = ACT_PATTERN.search(chunk)
                scene_match = SCENE_PATTERN.search(chunk)

                # a new act starts without scene until the next scene header
                if act_match:
                    current_act = to_roman(act_match.group(1))
                    current_scene = ""
                if scene_match:
                    current_scene = to_roman(scene_match.group(1))

                # section shown to the user: "Act I, Scene III"
                section = ""
                if current_act:
                    section = f"Act {current_act}"
                if current_scene:
                    section = section + f", Scene {current_scene}"
            else:
                # the first number alone on a line gives the poem number, kept until the next one
                for line in chunk.split("\n"):
                    number_match = number_line_pattern.match(line)
                    if number_match:
                        current_number = number_match.group(1)
                        break

                # section shown to the user: "Sonnet 18" or "Poem II"
                poem_label = "Sonnet" if "SONNET" in work["title"] else "Poem"
                section = ""
                if current_number:
                    section = f"{poem_label} {current_number}"

            # append the chunk and its metadata to the respective lists
            texts.append(chunk)
            ids.append("ID" + str(chunk_id))
            metadatas.append({
                "source": SOURCE,
                "chunk_id": chunk_id,
                "work": title,
                "type": work_type,
                "act": current_act,
                "scene": current_scene,
                "section": section
            })
            chunk_id += 1

    catalogue = build_catalogue(works)

    print(f"Total chunks: {len(texts)}")
    return texts, ids, metadatas, catalogue


def embed_and_store(texts, ids, metadatas):
    """
    Generates embeddings and upserts all documents into ChromaDB
    Loads a local embedding model to encode all the chunk texts, metadatas and IDs
    Uploads into the ChromaDB collection

    Args:
        texts (list[str]): chunk texts
        ids (list[str]): unique IDs for each chunk
        metadatas (list[dict]): metadata for each chunk
    """

    # creating the chromadb client and collection
    chromadb_client = chromadb.PersistentClient(path=str(CHROMA_PATH)) 

    collection = chromadb_client.get_or_create_collection(name=COLLECTION_NAME) 
    collection.delete(where={"source": SOURCE})  # wipes old chunks to avoid duplicates or lost chunks

    # ChromaDB searches the closest chunks approximately to go faster. With 12,500 chunks, a higher precision
    # makes the search give the same results at every run (checked on the test questions of rag_evaluation.ipynb)
    collection.modify(configuration={"hnsw": {"ef_search": SEARCH_PRECISION}})

    # We use a local model to avoid API calls: it is downloaded once and saved in the models folder
    if not MODEL_PATH.exists():
        SentenceTransformer(MODEL_NAME).save(str(MODEL_PATH))

    # load the model from local path
    model = SentenceTransformer(str(MODEL_PATH))

    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    """
    # tested and not kept: adding the work, act and scene before each chunk before embedding ("contextual" chunks)
    # the label takes too much room in short chunks, and the metadata filter already finds the right play
    texts_to_embed = [f"{meta['work']}, {meta['section']}\n{text}" for text, meta in zip(texts, metadatas)]
    embeddings = model.encode(texts_to_embed, batch_size=64, show_progress_bar=True).tolist()
    """

    # adding content to chromadb collection, in batches
    for start in range(0, len(texts), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
            embeddings=embeddings[start:end],
        )

    # check what is actually stored in ChromaDB with a small sample (5 chunks) to confirm the upsert worked and the metadata is correct
    sample = collection.get(limit=5)
    for doc, meta in zip(sample['documents'], sample['metadatas']):
        print(f"Meta: {meta}")
        print(f"Doc preview: {doc[:50]}")
        print("---")
    print(f"ChromaDB completed: {collection.count()} chunks")


if __name__ == "__main__":
    text = load_text()
    works = split_into_works(text)
    texts, ids, metadatas, catalogue = build_metadata(works)

    # save the catalogue of works and names for chat.py
    with open(CATALOGUE_PATH, "w", encoding="utf-8") as f:
        json.dump(catalogue, f, indent=2, ensure_ascii=False)

    embed_and_store(texts, ids, metadatas)
