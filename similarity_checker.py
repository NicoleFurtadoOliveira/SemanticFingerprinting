from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
from joblib import Parallel, delayed

# Function to load, clean, and compute TF-IDF matrix, ensuring it's cached only once
@st.cache_resource
def prepare_data():
    # Load dataset with target labels
    raw_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    # Clean text
    def clean_data(text):
        text = text.strip()
        text = re.sub(r'\n+', ' ', text)  # Replace paragraph breaks with a single space
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)  # Keep letters, numbers, spaces, and basic punctuation
        return text

    # Parallel data cleaning
    data = Parallel(n_jobs=-1)(delayed(clean_data)(doc) for doc in raw_data.data)

    # Compute TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data)
    
    return data, tfidf_matrix, raw_data.target, raw_data.target_names

# Load cleaned data, TF-IDF matrix, labels, and categories
data, tfidf_matrix, labels, categories = prepare_data()

# Function to calculate similarity and cache results
@st.cache_data
def calculate_similarity(doc1_index, doc2_index):
    similarity = cosine_similarity(tfidf_matrix[doc1_index], tfidf_matrix[doc2_index])
    return similarity[0][0]

# Streamlit UI
st.title("20 Newsgroups Document Similarity Checker")
st.write("Choose categories and document indexes to compare their similarity based on TF-IDF.")

## Dropdown for selecting categories
category1 = st.selectbox("Select Category for Document 1", categories, key="category1")
category2 = st.selectbox("Select Category for Document 2", categories, key="category2")

# Filter document indexes by selected categories
category1_indexes = [i for i, label in enumerate(labels) if categories[label] == category1]
category2_indexes = [i for i, label in enumerate(labels) if categories[label] == category2]

# Display valid document indexes within the selected categories
index1 = st.selectbox(f"Select Document Index for Category '{category1}'", category1_indexes, key="index1")
index2 = st.selectbox(f"Select Document Index for Category '{category2}'", category2_indexes, key="index2")

# Display "Calculate Similarity" button
if st.button("Calculate Similarity"):
    with st.spinner("Calculating similarity..."):
        # Show similarity score and documents 
        similarity_score = calculate_similarity(index1, index2)
        st.write(f"**Similarity Score:** {similarity_score:.4f}")

        st.subheader(f"Document 1 - {category1}")
        st.write(data[index1][:2000] + "..." if len(data[index1]) > 2000 else data[index1])

        st.subheader(f"Document 2 - {category2}")
        st.write(data[index2][:2000] + "..." if len(data[index2]) > 2000 else data[index2])
