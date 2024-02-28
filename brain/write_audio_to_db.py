import os
import sys
import torch
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, Float, String, JSON, DateTime
from transcription import Audio

engine = create_engine('postgresql+psycopg2://ashrithj:holdon@127.0.0.1:5432/cortex')

Base = declarative_base()

class User(Base):
    __tablename__ = 'speech'

    id = Column(Integer, primary_key=True)
    speaker = Column(String)
    start_time = Column(Float)
    end_time = Column(Float)
    spoken_text = Column(String)

Base.metadata.create_all(engine)
Base.metadata.bind = engine
Session = sessionmaker(bind=engine)

def add_element(element):
    session = Session()
    new_user = User(start_time=element.start, end_time=element.end, spoken_text=element.text)
    session.add(new_user)
    session.commit()
    session.close()


if __name__ == "__main__":
    audio_path = "./first_person_audio.wav"
    if sys.argv[1] == "transcribe":
        segments = Audio.transcript_audio(audio_path)
        for segment in segments:
            add_element(segment)

    session = Session()
    users = session.query(User).all()
    for user in users:
        print(f"ID: {user.id}, start: {user.start_time}, end: {user.end_time}, text: {user.spoken_text}")
    session.close()

