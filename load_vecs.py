'''
Just a sanity check to ensure that everything loads properly
'''
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, PickleType

engine = create_engine('sqlite:///database.db')
Base = declarative_base()
Base.metadata.bind = engine
DBSession = sessionmaker()
DBSession.bind = engine
session = DBSession()


class Vector(Base):
    __tablename__ = 'Vectors'
    id = Column(Integer, primary_key=True)
    rating = Column(Integer, nullable=False)
    clean_text = Column(
        String(5000),
        nullable=False)  # Max 3812 in dataset
    vector = Column(PickleType, nullable=False)

first = session.query(Vector).first()
print(first.id)
print()
print(first.rating)
print()
print(first.clean_text)
print()
print(first.vector)
