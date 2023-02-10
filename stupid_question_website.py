import streamlit as st
import pickle
import spacy
import numpy as np
import pandas as pd

nlp = spacy.load('en_core_web_md')
def spacy_lemmatize(text):
    ex = nlp(text)
    return ' '.join([token.lemma_ for token in ex if token.pos_ not in ['AUX', 'PUNCT', 'PRON', 'URL', 'NUM']])

def load_model():
    with open('/Users/winston/Documents/dsir-1031/projects/project_3/output/best_model.pkl', 'rb') as f:
        the_model = pickle.load(f)
    return the_model

stupid_question_dict = {'AskReddit': 'No',
                        'NoStupidQuestions': 'Yes'}

model = load_model()
    
st.title('Is this a stupid question?')

your_text = st.text_input('Possibly stupid question: ', max_chars=500)

if len(your_text) >= 5:
    text = spacy_lemmatize(your_text)
    df =pd.DataFrame({'spacy': text, 'spacy_unique_word_count': len(text.split())}, index=[0,1])
    st.write(stupid_question_dict[model.predict(df)[0]])