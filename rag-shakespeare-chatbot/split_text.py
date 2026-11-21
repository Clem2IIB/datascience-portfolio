# Text -> chunks: reads the complete works, splits them into works, then cuts each work into chunks
# fill_db.py uses these functions to build the metadata and fill the chroma database

from pathlib import Path
import re

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "pg100.txt" # Project Gutenberg eBook #100: the complete works of Shakespeare

# Chunking settings
MIN_LENGTH = 200 # chunks shorter than this are merged with the next one
MAX_LENGTH = 800 # chunks longer than this are cut, MiniLM only reads about 256 tokens (around 1,000 characters)

# Thresholds used to find the works
TITLE_MIN_OVERLAP = 0.5 # share of words a line must have in common with a title to be taken as the start of that work
MIN_BLANK_LINES_BEFORE_TITLE = 3 # in the file every work title follows at least 3 empty lines

# Patterns used to read the text (used both to cut the chunks here and to build the metadata in fill_db.py)
# Act header: "ACT I", "ACT V." or "ACT III. SCENE I. The palace"
ACT_PATTERN = re.compile(r"^(?:ACT|Act) ([IVXL]+|\d+)\.?(?:$|\s+(?:SCENE|Scene)\s)", re.MULTILINE)
# Scene header: "SCENE I. Venice.", "Scene X.", "SCENE 4." or "ACT V. SCENE 2."
SCENE_PATTERN = re.compile(r"^(?:(?:ACT|Act) [IVXL]+\.?\s+)?(?:SCENE|Scene) ([IVXL]+|\d+)\.", re.MULTILINE)
# Speaker name: a name in capitals followed by a full stop at the start of a line
# The file mixes two editions: the name is alone on its line ("IAGO.") or starts the speech ("  DUKE. Escalus!")
SPEAKER_PATTERN = re.compile(r"^\s*(?!ACT |SCENE )([A-Z][A-Z’' ]+)\.(?:\s|$)", re.MULTILINE)


def normalize(text):
    """
    Puts a text in capitals and keeps only letters, digits and single spaces, so titles and names can be compared
    "Othello, the Moor of Venice" -> "OTHELLO THE MOOR OF VENICE"

    Args:
        text (str): any text (a title, a name, a question)

    Returns:
        str: the normalized text
    """
    text = text.upper()
    text = re.sub(r"[^A-Z0-9 ]", " ", text) # punctuation and accents become spaces
    words = text.split() # split() also removes the extra spaces
    return " ".join(words)


def load_text():
    """Loads the complete works and keeps only the text between the Project Gutenberg start and end markers"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # cut the Project Gutenberg licence at the start and at the end of the file
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    start = text.find("\n", start) + 1 # the text starts on the line after the start marker
    text = text[start:end]

    return text


def read_table_of_contents(lines):
    """
    Reads the table of contents at the top of the file: the indented lines after the "Contents" line

    Args:
        lines (list[str]): the lines of the complete works

    Returns:
        tuple: (titles, contents_end): the list of work titles in order, and the line number where the contents end
    """
    titles = []
    contents_end = 0
    in_contents = False
    for i, line in enumerate(lines):
        if line.strip() == "Contents": # if the line is "Contents" (not indented), the next lines are the titles
            in_contents = True
        elif in_contents and line.startswith("   "): # if the line is indented, it is a title
            titles.append(line.strip())
        elif in_contents and line.strip() != "":
            # the first line that is not indented ends the contents
            contents_end = i
            break

    return titles, contents_end


def split_into_works(text):
    """
    Splits the complete works into one text per work (play or poem)
    The titles are read from the table of contents at the top of the file, then each title is searched in the text, in the same order
    The titles in the text do not always match the table of contents ("THE TRAGEDY OF MACBETH" is only "MACBETH" in the text),
    so a line counts as a title when it follows at least 3 empty lines and shares enough words with the expected title

    Args:
        text (str): the complete works without the Project Gutenberg licence

    Returns:
        list[dict]: one dict per work with its title (as written in the text) and its text
    """
    lines = text.split("\n")

    # 1. Read the table of contents to know which titles to look for, and in which order
    contents_titles, contents_end = read_table_of_contents(lines)

    # 2. Walk through the text and look for the next expected title
    starts = [] # (line number, title as written in the text)
    blank_lines = MIN_BLANK_LINES_BEFORE_TITLE # number of empty lines just before the current line (the first work comes right after the contents)
    for i in range(contents_end, len(lines)):
        line = lines[i].strip()
        if line == "":
            blank_lines += 1
            continue

        # a title candidate is a line in capitals that follows several empty lines
        is_candidate = blank_lines >= MIN_BLANK_LINES_BEFORE_TITLE and line == line.upper()
        blank_lines = 0

        if is_candidate and len(starts) < len(contents_titles):
            expected_title = contents_titles[len(starts)]
            expected_words = set(normalize(expected_title).split())
            line_words = set(normalize(line).split())

            # word overlap: share of the words of the shorter title that are also in the other one
            common_words = expected_words & line_words
            overlap = len(common_words) / min(len(expected_words), len(line_words))
            if overlap >= TITLE_MIN_OVERLAP:
                starts.append((i, line.rstrip(":")))

    # stop here if a work was not found, it means the file is laid out differently and the rules above must be checked
    if len(starts) != len(contents_titles):
        missing = contents_titles[len(starts):]
        raise ValueError(f"Found {len(starts)} works out of {len(contents_titles)}, not found: {missing}")

    # 3. Cut the text between two consecutive titles (the last work goes to the end of the file)
    works = []
    for n, (line_number, title) in enumerate(starts):
        if n + 1 < len(starts):
            next_start = starts[n + 1][0]
        else:
            next_start = len(lines)
        works.append({
            "title": title,
            "text": "\n".join(lines[line_number + 1:next_start])
        })

    print(f"Works found: {len(works)}")
    return works


def is_contents_block(chunk):
    """
    Tells if a chunk is the table of contents of a play: a block of several lines that are all act or scene headers
    In the real text an act or scene header is alone in its block, so such a block has no useful text

    Args:
        chunk (str): one block of text

    Returns:
        bool: True if the block only lists acts and scenes
    """
    # other headers found in the contents of some plays
    other_headers = ("Prologue", "Chorus", "Epilogue", "Induction")

    lines = chunk.split("\n")
    if len(lines) == 1:
        return False

    # If the block has multiple lines, check each line
    for line in lines:
        is_header = ACT_PATTERN.match(line) or SCENE_PATTERN.match(line) or line.strip(" .") in other_headers
        if not is_header:
            return False
    return True


def cut_long_chunk(chunk):
    """
    Cuts a chunk longer than MAX_LENGTH (mostly a long speech) into pieces of about the same size
    The cut is made between two lines of verse, and each piece starts again with the speaker's name

    Args:
        chunk (str): a chunk longer than MAX_LENGTH characters

    Returns:
        list[str]: the pieces of the chunk
    """
    # number of pieces needed, then the length each piece aims for (equal pieces avoid a tiny last piece)
    n_pieces = len(chunk) // MAX_LENGTH + 1
    target_length = len(chunk) / n_pieces

    pieces = []
    speaker = "" # last speaker name alone on its line ("HAMLET."), repeated at the start of the next piece
    piece = ""
    for line in chunk.split("\n"):
        # remember the line if it is only a speaker name
        speaker_match = SPEAKER_PATTERN.match(line)
        if speaker_match and line.strip() == speaker_match.group(1) + ".":
            speaker = line

        piece = piece + line + "\n"

        # the piece is long enough: keep it and start the next one with the speaker name
        if len(piece) >= target_length:
            pieces.append(piece.strip())
            if speaker:
                piece = speaker + "\n"
            else:
                piece = ""

    # keep the last piece, unless it only holds the speaker name
    if piece.strip() and piece.strip() != speaker:
        pieces.append(piece.strip())

    return pieces


def chunk_text(text):
    """
    Splits the text into useful chunks on double newlines
    Merges consecutive chunks until they reach MIN_LENGTH characters
    Avoids very short chunks (like author stage directions or single lines) being stored as isolated passages without context
    Cuts the chunks longer than MAX_LENGTH on line breaks, so the embedding model reads the whole chunk

    Args:
        text (str): the text of one work

    Returns:
        list[str]: list of merged text chunks for embedding.
    """

    # Split with regex on double newlines
    raw_chunks = re.split(r'\n\n+', text)

    merged_chunks = [] # empty list to hold the final merged chunks
    current_chunk = "" # temporary variable to build up the current chunk being merged

    # iterate through the raw chunks and merge them until they reach the minimum length, then add to merged_chunks
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk or chunk == "Contents":
            continue

        # skip the table of contents of a play
        if is_contents_block(chunk):
            continue

        if len(current_chunk) == 0:
            current_chunk = chunk
        else:
            current_chunk = current_chunk + "\n\n" + chunk

        # Only close the chunk when it's long enough AND ends naturally
        if len(current_chunk) >= MIN_LENGTH:
            merged_chunks.append(current_chunk)
            current_chunk = ""

    # Add any remaining chunk that did not reach the minimum length but still has content
    # it is joined to the last chunk ("[_Exeunt._]" at the end of a play is not useful alone)
    if current_chunk and merged_chunks:
        merged_chunks[-1] = merged_chunks[-1] + "\n\n" + current_chunk
    elif current_chunk:
        merged_chunks.append(current_chunk)

    # Cut the chunks that are too long into pieces below MAX_LENGTH characters
    final_chunks = []
    for chunk in merged_chunks:
        if len(chunk) <= MAX_LENGTH:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(cut_long_chunk(chunk))

    # only plays have speakers: they are the works with acts, in the poems the capital lines are titles or numbers
    if ACT_PATTERN.search(text):
        final_chunks = add_missing_speakers(final_chunks)

    return final_chunks


def add_missing_speakers(chunks):
    """
    Adds the name of the last speaker at the start of a chunk that has no speaker name
    A long speech is sometimes split by a stage direction ("[_Laying Paris in the monument._]"), so its end
    becomes a chunk without the name of who speaks, which the search and the LLM cannot link to the character

    Args:
        chunks (list[str]): the chunks of one work, in the order of the text

    Returns:
        list[str]: the same chunks, with the speaker name added where it was missing
    """
    chunks_with_speakers = []
    last_speaker = "" # name of the last character who spoke, like "ROMEO"
    for chunk in chunks:
        speakers = SPEAKER_PATTERN.findall(chunk)
        headers = ACT_PATTERN.findall(chunk) + SCENE_PATTERN.findall(chunk)

        # no speaker and no new act or scene: the chunk continues the speech of the last speaker
        if not speakers and not headers and last_speaker:
            chunk = last_speaker + ".\n" + chunk
        chunks_with_speakers.append(chunk)

        # remember who speaks last, a new act or scene without any speaker starts with nobody speaking
        if speakers:
            last_speaker = speakers[-1].strip()
        elif headers:
            last_speaker = ""

    return chunks_with_speakers

"""
# previous technique for splitting the document with overlap, but we leaned towards a more dynamic approach based on natural breaks in the text (double newlines) to better preserve context and metadata

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100, #Nb of characters shared between two consecutive chunks
    length_function=len,
    is_separator_regex=False
)
chunks = text_splitter.split_text(othello)
"""
