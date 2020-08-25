import nltk
import sys
import os
import string
import math


FILE_MATCHES = 1

def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    corpus = dict()
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if name.endswith('.txt'):
                with open(os.path.join(path, name)) as file:
                    key = name[0:-4]
                    value = file.read()
                    corpus[key] = value
    return corpus

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punctuation = []
    for character in string.punctuation:
        punctuation.append(character)
    stopwords = nltk.corpus.stopwords.words("english")
    # Note: This keeps the word "n't". To also remove this word,
    # we could add "n't' to the list of stopwords."
    words = [
            word.lower() for word in
            nltk.word_tokenize(document)
        ]
    words = [word for word in words if word not in punctuation and word not in stopwords]
    return words

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Get all words in corpus
    words = set()
    for filename in documents:
        words.update(documents[filename])
    # Calculate IDFs
    idfs = dict()
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Count frequencies
    frequencies = dict()
    for filename in files:
        text = files[filename]
        words_counter = dict()
        for word in text:
            if word not in words_counter:
                words_counter[word] = 1
            else:
                words_counter[word] += 1
        frequencies[filename] = words_counter
    # Calculate tfidfs
    tfidfs = dict()
    for filename in frequencies:
        tfidfs[filename] = []
        for word in frequencies[filename]:
            tf = frequencies[filename][word]
            tfidfs[filename].append((word, tf * idfs[word]))
    # Determine list of the filenames of the n top files 
    relevant_tfidfs = dict()
    for filename in tfidfs:
        value = 0
        for entry in tfidfs[filename]:
            if list(entry)[0] in query:
                value += list(entry)[1]
        relevant_tfidfs[filename] = value
    result = sorted(relevant_tfidfs, key=relevant_tfidfs.get, reverse=True)
    return result[0:n]

directory = 'corpus_test'
files = load_files(directory)
file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
file_idfs = compute_idfs(file_words)
query = set(('sanctioned', 'machine'))
filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

# Extract sentences from top files
sentences = dict()
for filename in filenames:
    for passage in files[filename].split("\n"):
        for sentence in nltk.sent_tokenize(passage):
            tokens = tokenize(sentence)
            if tokens:
                sentences[sentence] = tokens

# Compute IDF values across sentences
idfs = compute_idfs(sentences)

# Determine top sentence matches
#matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)

relevant_idfs = dict()
for sentence, tokens in sentences.items():
    value = 0
    for word, idf_score in idfs.items():
        if word in query and word in tokens:
            value += idf_score
    relevant_idfs[sentence] = value

print(relevant_idfs)

