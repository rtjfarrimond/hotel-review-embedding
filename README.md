# Hotel reviews embedding space
In this repo we define a doc2vec model to map reviews from the [Kaggle Hotel Reviews dataset](https://www.kaggle.com/datafiniti/hotel-reviews) to vecotrs in an embedding space, then persist these vectors in a SQLite database.

## Requirements
### All
- python 3.7
- pandas
- numpy
- sqlite3
- gensim
- langdetect
- sqlalchemy
- nltk

### Notebook
- matplotlib
- seaborn
- scikit-learn

## Train and evaluate model.ipynb
In this notebook the data is examined and a doc2vec model is trained, evaluated and saved as **doc2vec.mdl**.

## pipeline.py
This script defines a pipeline to process the data in the same fashion as in the jupyter notebook, uses doc2vec.mdl to define vectors for each review, and saves these to a SQLite database. Usage:

`$ python pipeline.py <data path> <model path> <db path>`

## database.db
This is the database where the cleaned data and associated vectors are persisted. To query using command line:

`sqlite3 database.db`

## load_vecs.py
This is just a script to sanity check that data from the db can be properly recovered by querying.
