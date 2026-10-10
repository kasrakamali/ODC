import json
import pathlib
import pickle
import re
import zipfile
from collections import Counter
from typing import Iterable, List, Optional, Set, Union

import ftfy
import pandas as pd
import requests
import unidecode
from tqdm.auto import tqdm

DEFAULT_OPENIMAGES_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
DEFAULT_VG_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip"
TWO_LETTER_WHITELIST = {"tv", "pc", "vr", "xr", "3d"}


def normalize_text(text: str) -> str:
    """Normalizes text: unidecode, lowercase, strips weird punctuation."""
    text = ftfy.fix_text(unidecode.unidecode(str(text).lower()))
    text = re.sub(r"[^0-9a-z\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_good_visual_term(term: str, stop_words: Set[str]) -> bool:
    """Heuristic filter for Visual Genome & Open Images tags."""
    if len(term) < 2 or len(term) > 30:
        return False
    words = term.split()
    if not (1 <= len(words) <= 3):
        return False
    if not re.search(r"[a-z]", term):
        return False
    if sum(ch.isdigit() for ch in term) > len(term) * 0.4:
        return False
    if len(term) == 2 and term not in TWO_LETTER_WHITELIST:
        return False
    if re.fullmatch(r"(number|object|img|image|item)\s+\d+[a-z]*", term):
        return False
    if re.fullmatch(r"\d+", term):
        return False
    if all(w in stop_words for w in words):
        return False
    return True


def is_good_text_term(term: str, stop_words: Set[str]) -> bool:
    """Strict clean filter for text sources (Base dictionary & Wikipedia)."""
    if len(term) < 3:
        return False
    if any(char.isdigit() for char in term):
        return False
    if not (term[0].isalpha() and term[-1].isalpha()):
        return False
    if not re.fullmatch(r"[a-z\s'-]+", term):
        return False
    words = term.split()
    if len(words) > 1 and all(w in stop_words for w in words):
        return False
    return True


def download_file(url: str, destination: pathlib.Path, chunk_size: int = 1024 * 1024, verbose: bool = True) -> None:
    """Downloads a remote file with progress bar if it doesn't already exist."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if verbose:
            print(f"Using cached file: {destination.name}")
        return

    if verbose:
        print(f"Downloading from {url}...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=destination.name, disable=not verbose
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def get_stopwords() -> Set[str]:
    """Gets English stopwords from NLTK."""
    import nltk
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))


def get_spacy_pipeline(model_name: str = "en_core_web_sm"):
    """Loads spaCy model with NER disabled for fast parsing."""
    import spacy
    try:
        nlp = spacy.load(model_name, disable=["ner"])
    except OSError:
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name, disable=["ner"])
    nlp.max_length = 2_000_000
    return nlp


def extract_openimages_labels(
    cache_dir: Union[str, pathlib.Path] = "./cache_data",
    url: str = DEFAULT_OPENIMAGES_URL,
    verbose: bool = True,
) -> Set[str]:
    cache_dir = pathlib.Path(cache_dir)
    csv_path = cache_dir / "openimages_boxable.csv"
    download_file(url, csv_path, verbose=verbose)

    df = pd.read_csv(csv_path, header=None, names=["id", "label"])
    labels = {normalize_text(x) for x in df["label"].dropna()}
    if verbose:
        print(f"Open-Images: extracted {len(labels):,} labels")
    return labels


def extract_visual_genome_labels(
    cache_dir: Union[str, pathlib.Path] = "./cache_data",
    url: str = DEFAULT_VG_URL,
    verbose: bool = True,
) -> Set[str]:
    cache_dir = pathlib.Path(cache_dir)
    zip_path = cache_dir / "vg_objects.zip"
    download_file(url, zip_path, verbose=verbose)

    if verbose:
        print("Unzipping & loading Visual Genome JSON (~2 GB RAM)...")
    with zipfile.ZipFile(zip_path) as z:
        with z.open("objects.json") as f:
            vg_data = json.load(f)

    names = set()
    for img in tqdm(vg_data, desc="Visual Genome", disable=not verbose):
        for obj in img.get("objects", []):
            for n in obj.get("names", []):
                names.add(normalize_text(n))

    if verbose:
        print(f"Visual Genome: extracted {len(names):,} labels")
    return names


def extract_base_dictionary(verbose: bool = True) -> Set[str]:
    try:
        from english_words import get_english_words_set
        words = get_english_words_set(["web2"], lower=True)
        if verbose:
            print(f"Base dictionary: loaded {len(words):,} words")
        return set(words)
    except ImportError:
        print("Warning: 'english-words' package not found. Skipping base dictionary.")
        return set()


def extract_wikipedia_phrases(
    num_articles: int = 1000,
    min_freq: int = 3,
    spacy_model: str = "en_core_web_sm",
    verbose: bool = True,
) -> Set[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: 'datasets' package not found. Skipping Wikipedia.")
        return set()

    nlp = get_spacy_pipeline(spacy_model)
    if verbose:
        print(f"Streaming {num_articles} Wikipedia articles...")
    wiki_stream = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    counts = Counter()
    for article in tqdm(
        wiki_stream.take(num_articles),
        total=num_articles,
        desc="Wikipedia phrases",
        disable=not verbose,
    ):
        text = article.get("text", "")
        if not text.strip():
            continue
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            if len(chunk) > 1:
                counts[chunk.text.lower().strip()] += 1

    extracted = {phrase for phrase, count in counts.items() if count >= min_freq}
    if verbose:
        print(f"Wikipedia: kept {len(extracted):,} phrases (freq >= {min_freq})")
    return extracted


def spell_check_phrases(phrases: Iterable[str], verbose: bool = True) -> Set[str]:
    try:
        from symspellpy import SymSpell
    except ImportError:
        print("Warning: 'symspellpy' not found. Skipping spell correction.")
        return set(phrases)

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    try:
        from importlib import resources
        dict_path = resources.files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt")
        sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1)
    except Exception:
        import pkg_resources
        dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

    unique_phrases = list(set(phrases))
    corrected = set()
    for phrase in tqdm(unique_phrases, desc="Spell checking", disable=not verbose):
        suggestions = sym_spell.lookup_compound(phrase, max_edit_distance=2)
        corrected.add(suggestions[0].term if suggestions else phrase)

    return corrected


def build_vocabulary(
    use_openimages: bool = True,
    use_visual_genome: bool = True,
    use_spellcheck: bool = True,
    use_base_dict: bool = True,
    use_wikipedia: bool = True,
    num_wiki_articles: int = 1000,
    min_wiki_freq: int = 3,
    cache_dir: str = "./cache_data",
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Builds, filters, and combines a multi-source vocabulary.
    Returns the vocabulary as a sorted list of strings.
    """
    cache_path = pathlib.Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    stop_words = get_stopwords()

    # Step 1: Visual Vocab
    visual_vocab = set()
    if use_openimages:
        visual_vocab |= extract_openimages_labels(cache_path, verbose=verbose)
    if use_visual_genome:
        visual_vocab |= extract_visual_genome_labels(cache_path, verbose=verbose)

    if visual_vocab:
        if verbose:
            print(f"Filtering {len(visual_vocab):,} visual terms...")
        visual_vocab = {t for t in visual_vocab if is_good_visual_term(t, stop_words)}

        if use_spellcheck:
            if verbose:
                print("Running spell-checking...")
            visual_vocab = spell_check_phrases(visual_vocab, verbose=verbose)

    # Step 2: Textual Vocab
    textual_vocab = set()
    if use_base_dict:
        textual_vocab |= extract_base_dictionary(verbose=verbose)
    if use_wikipedia:
        textual_vocab |= extract_wikipedia_phrases(
            num_articles=num_wiki_articles,
            min_freq=min_wiki_freq,
            verbose=verbose,
        )

    if textual_vocab:
        if verbose:
            print(f"Filtering {len(textual_vocab):,} textual terms...")
        textual_vocab = {t for t in textual_vocab if is_good_text_term(t, stop_words)}

    # Step 3: Combine
    final_vocab = sorted(visual_vocab | textual_vocab)
    if verbose:
        print(f"\n🎉 Done! Final vocabulary size: {len(final_vocab):,} unique items")

    # Step 4: Optional Save
    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ext = out.suffix.lower()

        if ext == ".pkl":
            with open(out, "wb") as f:
                pickle.dump(final_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif ext == ".json":
            with open(out, "w", encoding="utf-8") as f:
                json.dump(final_vocab, f, indent=2, ensure_ascii=False)
        else:
            with open(out, "w", encoding="utf-8") as f:
                for word in final_vocab:
                    f.write(f"{word}\n")

        if verbose:
            print(f"Saved vocabulary to: {out.resolve()}")

    return final_vocab
