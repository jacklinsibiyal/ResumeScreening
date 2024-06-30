import altair as alt
import datetime
import nltk
import numpy as np
import pandas as pd
import re
import streamlit as st 
import time
import io
import pdfkit
import docx2txt
import subprocess
import PyPDF2  # Required for reading PDF files
import docx  # Required for reading DOCX files
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from PyPDF2.errors import PdfReadError
from gensim.models import KeyedVectors, TfidfModel
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
from gensim.similarities.annoy import AnnoyIndexer
from io import BytesIO
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pandas.api.types import is_numeric_dtype
from keras.src.utils.numerical_utils import to_categorical
import pytesseract
from PIL import Image
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.src.saving.saving_api import load_model
import pickle
from keras.src.utils.sequence_utils import pad_sequences



nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def read_resumes(file):
    resumes = []

    if file.type == 'application/pdf':
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        for page_num in range(len(pdf_reader.pages)):
            text = pdf_reader.pages[page_num].extract_text()
            resumes.append(text)

    elif file.type in ['image/png', 'image/jpeg', 'image/jpg']:
        image = Image.open(io.BytesIO(file.read()))
        text = image_to_text(image)
        resumes.append(text)

    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        text = docx2txt.process(io.BytesIO(file.read()))
        resumes.append(text)

    elif file.type == 'text/plain':
        text = file.read().decode('utf-16', 'ignore')
        resumes.append(text)

    elif file.type == 'application/msword':
        # If textract cannot handle DOC files, consider alternatives like antiword:
        # text = textract.process(file.read())
            # Assuming antiword is installed
        process = subprocess.Popen(['antiword', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output, _ = process.communicate(file.read())
        text = output.decode('utf-8')
        resumes.append(text)


    elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        excel_df = pd.read_excel(io.BytesIO(file.read()))
        for column in excel_df.columns:
            resumes.extend(excel_df[column].astype(str))

    return pd.DataFrame({'Resume': resumes})

def image_to_text(image):
    # Convert image to text using OCR (Optical Character Recognition)
    text = pytesseract.image_to_string(image)
    return text

def clickRank():
    st.session_state.processRank = True

def convertDfToXlsx(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
    df.to_excel(writer, index = False, sheet_name = 'Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processedData = output.getvalue()
    return processedData


def getWordnetPos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def performLemmatization(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    text = re.sub('\s+', ' ', text)
    words = word_tokenize(text)
    words = [
        lemmatizer.lemmatize(word.lower(), pos = getWordnetPos(pos)) 
        for word, pos in pos_tag(words) if word.lower() not in stop_words
    ]
    return words

def performStemming(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    text = re.sub('\s+', ' ', text)
    words = word_tokenize(text)
    words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    text = ' '.join(words)
    return text 

def loadModel():
    # Load the model
    model = load_model('resumescreening.h5')
    return model

model = loadModel()

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def rankResumes(text, df):
    progressBar = st.progress(0)
    progressBar.progress(0, text="Preprocessing data ...")
    startTime = time.time()
    jobDescriptionText = performLemmatization(text)
    df['cleanedResume'] = df['Resume'].apply(lambda x: performLemmatization(x))
    documents = [jobDescriptionText] + df['cleanedResume'].tolist()
    progressBar.progress(13, text="Creating a dictionary ...")
    dictionary = Dictionary(documents)
    progressBar.progress(25, text="Creating a TF-IDF model ...")
    tfidf = TfidfModel(dictionary=dictionary)
    progressBar.progress(38, text="Calculating TF-IDF vectors...")
    tfidf_vectors = tfidf[[dictionary.doc2bow(resume) for resume in df['cleanedResume']]]
    query_vector = tfidf[dictionary.doc2bow(jobDescriptionText)]

    progressBar.progress(50, text="Calculating similarity scores...")
    similarities = []
    for vector in tfidf_vectors:
        # Convert query_vector and vector to numpy arrays
        query_vector_array = np.array(query_vector)
        vector_array = np.array(vector)
        
        # Ensure both arrays have the same number of dimensions
        if query_vector_array.ndim != vector_array.ndim:
            vector_array = np.expand_dims(vector_array, axis=0)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(query_vector_array, vector_array)
        similarities.append(similarity[0][0])

    progressBar.progress(75, text="Ranking resumes...")
    df['Similarity Score (-1 to 1)'] = similarities
    df['Rank'] = df['Similarity Score (-1 to 1)'].rank(ascending=False, method='dense').astype(int)
    df.sort_values(by='Rank', inplace=True)
    df.drop(columns=['cleanedResume'], inplace=True)
    
    endTime = time.time()
    elapsedSeconds = endTime - startTime
    hours, remainder = divmod(int(elapsedSeconds), 3600)
    minutes, _ = divmod(remainder, 60)
    secondsWithDecimals = '{:.2f}'.format(elapsedSeconds % 60)
    elapsedTimeStr = f'{hours} h : {minutes} m : {secondsWithDecimals} s'
    
    progressBar.progress(100, text=f'Ranking Complete!')
    time.sleep(1)
    progressBar.empty()
    
    st.info(f'Finished ranking {len(df)} resumes - {elapsedTimeStr}')
    return df 

def clickClassify():
    st.session_state.processClf = True

def addZeroFeatures(matrix):
    maxFeatures = 18038
    numDocs, numTerms = matrix.shape
    missingFeatures = maxFeatures - numTerms
    if missingFeatures > 0:
        zeroFeatures = csr_matrix((numDocs, missingFeatures), dtype=np.float64)
        matrix = hstack([matrix, zeroFeatures])
    return matrix

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace

    words = resumeText.split()
    words = [word for word in words if word.lower() not in stop_words]
    words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    resumeText = ' '.join(words)
    return resumeText

def createBarChart(df):
    valueCounts = df['Industry Category'].value_counts().reset_index()
    valueCounts.columns = ['Industry Category', 'Count']
    newDataframe = pd.DataFrame(valueCounts)
    barChart = alt.Chart(newDataframe,
    ).mark_bar(
        color = '#56B6C2',
        size = 13 
    ).encode(
        x = alt.X('Count:Q', axis = alt.Axis(format = 'd'), title = 'Number of Resumes'),
        y = alt.Y('Industry Category:N', title = 'Category'),
        tooltip = ['Industry Category', 'Count']
    ).properties(
        title = 'Number of Resumes per Category',
    )
    return barChart

def classifyResumes(df):
    progressBar = st.progress(0)
    progressBar.progress(0, text="Preprocessing data ...")
    startTime = time.time()
    
    df['cleanedResume'] = df['Resume'].apply(lambda x: performStemming(x))
    resumeText = df['cleanedResume'].values
    
    # Load the model
    model = load_model('resumescreening.h5')

    # Load the tokenizer
    tokenizer = load_tokenizer('tokenizer.pickle')
    
    # Tokenize and pad the sequences
    sequences = tokenizer.texts_to_sequences(resumeText)
    padded_sequences = pad_sequences(sequences, maxlen=200, padding='post')

    num_resumes = padded_sequences.shape[0]  # Get the number of resumes
    sequence_length = padded_sequences.shape[1]  # Get the sequence length (should be 200)

    # Reshape padded_sequences to match the expected input shape for LSTM
    padded_sequences = padded_sequences.reshape((num_resumes, sequence_length, -1))
    # Check the shape of the padded_sequences
    st.write(f'Padded sequences shape: {padded_sequences.shape}')
    
    progressBar.progress(20, text="Extracting features ...")
    
    progressBar.progress(60, text="Predicting categories ...")
    
    # Predict categories using the loaded model
    predicted_probabilities = model.predict(padded_sequences)
    predicted_categories = predicted_probabilities.argmax(axis=1)

    
    progressBar.progress(80, text="Finishing touches ...")
    
    # Load the LabelEncoder object
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    # Assuming `df` is your DataFrame
    df['Industry Category'] = le.inverse_transform(predicted_categories)
    df['Industry Category'] = pd.Categorical(df['Industry Category'])
    df.drop(columns=['cleanedResume'], inplace=True)

    endTime = time.time()
    elapsedSeconds = endTime - startTime
    hours, remainder = divmod(int(elapsedSeconds), 3600)
    minutes, _ = divmod(remainder, 60)
    secondsWithDecimals = '{:.2f}'.format(elapsedSeconds % 60)
    elapsedTimeStr = f'{hours} h : {minutes} m : {secondsWithDecimals} s'
    
    progressBar.progress(100, text=f'Classification Complete!')
    time.sleep(1)
    progressBar.empty()
    st.info(f'Finished classifying {len(resumeText)} resumes - {elapsedTimeStr}')
    
    return df
