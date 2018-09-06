import pandas as pd
import string

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from langdetect import detect
from langdetect import DetectorFactory


def load_data(path):
    pass


def clean_data(data):
    DetectorFactory.seed = 0 # enforce deterministic language detection for short / ambiguous text
    pass


def define_vectors(data):
    pass


def insert_vectors(db, vectors):
    pass
