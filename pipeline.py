from __future__ import print_function
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from langdetect import detect
from langdetect import DetectorFactory
from nltk.corpus import stopwords
from sqlalchemy import create_engine, Column, Integer, String, PickleType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import pandas as pd
import string
import sqlite3
import sys
import traceback


def insert_vectors(data_path, model_path, db_path):
    '''
    Given a path to the csv data, a path to a pre-trained gensim doc2vec model,
    and a path to a SQLite db (which may or may not exist), this function will
    clean and process the data
    '''
    def prepare_data():
        data = pd.read_csv(data_path)
        model = Doc2Vec.load(model_path)

        print('Preparing data...')

        def clean_text(text):
            '''
            Replace invalid characters, lowercase everything,
            remove punctuation and whitespace.
            '''
            return text.replace('�', '').lower().translate(
                str.maketrans('', '', string.punctuation)).strip()

        def exception_safe_detect(s):
            '''
            langdetect.detect will throw an exception when it is unable to
            determine a language from input. We will drop these cases later
            so return empty string instead.
            '''
            try:
                return detect(s)
            except:
                return ''

        def append_vectors(tokens):
            '''
            Where tokens is a pandas Series, this function will apply
            the infer_vector function of the doc2vec loaded from model_path.
            '''
            model = Doc2Vec.load(model_path)
            return tokens.apply(model.infer_vector)

        def clean_stopwords(tokens, stopWords):
            ret = []
            for t in tokens:
                if t not in stopWords:
                    ret.append(t)

            return pd.Series(ret)

        # enforce deterministic language detection for short / ambiguous text
        DetectorFactory.seed = 0
        stopWords = set(stopwords.words('english'))

        # Drop columns
        cleaned = data[[
            'reviews.title', 'reviews.text', 'reviews.rating']].copy()

        # Remove dupes
        cleaned.drop_duplicates(
            inplace=True, subset='reviews.text', keep=False)
        cleaned.drop_duplicates(
            inplace=True, subset='reviews.title', keep=False)

        # Merge fields and rename
        cleaned['review'] = cleaned['reviews.title'] + \
            ' ' + cleaned['reviews.text']
        cleaned.drop(
            labels=['reviews.title', 'reviews.text'], inplace=True, axis=1)
        cleaned.rename(
            index=str, columns={'reviews.rating': 'rating'}, inplace=True)

        # Drop NaN
        cleaned = cleaned.dropna()

        # Remove noisy ratings
        cleaned = cleaned[cleaned['rating'].isin(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])]

        # Clean up the review text
        cleaned['review'] = cleaned['review'].apply(clean_text)

        # Remove non-English reviews
        print('Removing non-English reviews...')
        cleaned = cleaned[cleaned['review'].apply(
            exception_safe_detect) == 'en']

        # Remove dupes again before tokenising, since lists are unhashable
        cleaned.drop_duplicates(inplace=True, keep=False, subset='review')
        cleaned['review_tokens'] = cleaned['review'].str.split()

        # Remove stopwords
        # cleaned['review'] = cleaned['review'].apply(
        #     clean_stopwords, args=(stopWords,))

        # Compute the vectors
        print('Computing vectors...')
        cleaned['vector'] = append_vectors(cleaned['review_tokens'])

        return cleaned

    def insert(data):
        engine = create_engine(db_path)
        Base = declarative_base()
        Base.metadata.bind = engine

        print('Inserting data...')

        class Vector(Base):
            __tablename__ = 'Vectors'
            id = Column(Integer, primary_key=True)
            rating = Column(Integer, nullable=False)
            clean_text = Column(
                String(5000),
                nullable=False)  # Max 3812 in dataset
            vector = Column(PickleType, nullable=False)

        # If the Vectors table does not already exist, then create it
        if not engine.dialect.has_table(engine, 'Vectors'):
            Base.metadata.create_all(engine)

        DBSession = sessionmaker(bind=engine)
        session = DBSession()

        try:
            for row in data.itertuples():
                session.add(Vector(
                    rating=row[1],
                    clean_text=row[2],
                    vector=row[4]))

            session.commit()
            print('Data inserted successfully!')
        except:
            session.rollback()
            print('An excpection occurred. Transaction rolled back.\n')
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            

    insert(prepare_data())


if __name__ == '__main__':
    print('-' * 65)
    print('Usage: \t $ python pipeline.py <data path> <model path> <db path>')
    print('-' * 65)
    print()

    insert_vectors(sys.argv[1], sys.argv[2], 'sqlite:///' + sys.argv[3])
