import random
import os
import glob
import numpy as np
import string
import spacy
import pyttsx3
import hnswlib
import torch
import secrets
from scipy.sparse 
import issparseimport random
import os
import glob
import numpy as np
import string
import spacy
import pyttsx3
import hnswlib
import torch
import secrets
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from scipy.sparse import issparse
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import homogeneity_score
from sklearn.metrics import silhouette_score
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Secret key
secret_key = secrets.token_hex(16)

# Setting the secret key
app.secret_key = secret_key

nlp = spacy.load("en_core_web_md")

# Function to load data from transcript files
def load_data(folder_path):
    data = []
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data.append(line.strip())
    return data

# Function to preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def vectorize_data_with_embeddings(data):
    nlp = spacy.load("en_core_web_lg")
    vectors = [nlp(text).vector for text in data]
    return np.array(vectors), None

def calculate_similarity(user_question, bot_response):
    # Tokenize and vectorize the user question and bot response using spaCy
    doc1 = nlp(user_question)
    doc2 = nlp(bot_response)

    # Calculate cosine similarity between document vectors
    similarity = cosine_similarity([doc1.vector], [doc2.vector])[0][0]

    return similarity

def agglomerative_cluster_data(vectors, num_clusters=14):
    clustering_model = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
    clusters = clustering_model.fit_predict(vectors)
    return clusters, clustering_model

def calculate_cluster_homogeneity(clustering_model, vectors):
    labels_true = clustering_model.labels_
    if issparse(vectors):
        vectors = vectors.toarray()
    labels_pred = clustering_model.fit_predict(vectors)
    homogeneity = homogeneity_score(labels_true, labels_pred)
    return homogeneity

def calculate_silhouette_score(vectors, clustering_model):
    labels = clustering_model.labels_
    silhouette = silhouette_score(vectors, labels, metric='euclidean')
    return silhouette

# Function to get fallback responses for out-of-cluster queries
def get_fallback_response():
    fallback_responses = [
        "I'm sorry, but I don't have information on that topic. Please try asking something else.",
        "I'm not sure I understand. Could you please rephrase your question?",
        "I'm still learning, and I don't have an answer for that. Can you try a different question?",
    ]
    return random.choice(fallback_responses)

def get_intent(user_query, clustering_model, vectorizer=None):
    if vectorizer:
        user_query_vector = vectorizer.transform([user_query])
    else:
        user_query_vector = np.array([nlp(user_query).vector])
    # Get the cluster labels for the existing vectors
    cluster_labels = clustering_model.labels_
    # Convert user_query_vector to a dense array if needed
    if vectorizer:
        user_query_vector = user_query_vector.toarray()
    # Append the user query vector to the existing data
    vectors_extended = np.concatenate([vectors, user_query_vector], axis=0)  # Concatenate along axis 0
    user_cluster = clustering_model.fit_predict(vectors_extended)[-1]  # Last element is the user query
    # Restore the original cluster labels
    clustering_model.labels_ = cluster_labels
    return user_cluster

def context_matching(user_query, responses):
    tfidf_vectorizer = TfidfVectorizer()
    response_vectors = tfidf_vectorizer.fit_transform(responses)
    user_query_vector = tfidf_vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(user_query_vector, response_vectors)[0]
    sorted_indices = np.argsort(similarity_scores)[::-1]
    sorted_responses = [responses[i] for i in sorted_indices]
    return sorted_responses

# Function to generate speech output from text
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech (words per minute)
    engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
    engine.say("I will now start answering your question")
    engine.say(text)
    engine.runAndWait()

def vectorize_data_with_tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    vectors = tfidf_vectorizer.fit_transform(data)
    return vectors.toarray(), tfidf_vectorizer 

# Function to run the LLM-based chatbot
def run_llm_chatbot(question, output_preference):
    model_name = "static/distilgpt2_finetuned_model"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    model.eval()
    input_ids = tokenizer.encode(question, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_response

# Load and preprocess data
data_folder = "static/COMP3074transcripts"
data = load_data(data_folder)
preprocessed_data = [preprocess_text(item) for item in data]

# Vectorize data with TF-IDF
vectors, tfidf_vectorizer = vectorize_data_with_tfidf(preprocessed_data)

# Agglomerative Clustering
num_clusters = 14
clusters, clustering_model = agglomerative_cluster_data(vectors, num_clusters)

# Function to get or initialize the conversation history in the session
def get_or_init_conversation():
    if 'conversation' not in session:
        session['conversation'] = []  # Initialize an empty list for the conversation
    return session['conversation']

@app.route('/retrieval_based_chatbot', methods=['GET', 'POST'])
def retrieval_based_chatbot():
    # Initialize or retrieve the conversation history from the session
    conversation = session.get('conversation', [])
    
    if request.method == 'POST':
        # Get the user query and output preference from the form data
        user_query = request.form.get('user_query', '')
        output_preference = request.form.get('output_preference', 'text')

        # Pass the vectors and vectorizer as arguments to the get_response_helper function
        response = get_response_helper(user_query, data, clustering_model, vectors, tfidf_vectorizer)

        # Evaluation Metrics
        homogeneity = calculate_cluster_homogeneity(clustering_model, vectors)
        print("Cluster Homogeneity (Agglomerative Clustering):", homogeneity)
        
        # Calculate Silhouette Score
        silhouette_score = calculate_silhouette_score(vectors, clustering_model)
        print("Silhouette Score:", silhouette_score)

        # Update the conversation history with the user query and chatbot response
        conversation.append({'role': 'user', 'content': user_query})
        conversation.append({'role': 'chatbot', 'content': response})

        if output_preference == 'voice':
            # Call the speak function to generate speech output
            speak(response)

        # Save the updated conversation history back to the session
        session['conversation'] = conversation

        if output_preference == 'text':
            return render_template('retrieval_based_chatbot.html', conversation=conversation)
    
    # If it's a GET request or the output_preference is 'voice', render the template directly
    return render_template('retrieval_based_chatbot.html', conversation=conversation)

@app.route('/end_conversation', methods=['POST'])
def end_conversation():
    chatbot_type = request.form.get('chatbot_type', 'retrieval')  # Default to "retrieval" chatbot
    if chatbot_type == 'retrieval':
        # Clear the retrieval-based chatbot's conversation history from the session
        session.pop('conversation', None)
    elif chatbot_type == 'llm':
        # Clear the LLM-based chatbot's conversation history from the session
        session.pop('llm_conversation', None)
    return redirect(url_for(f'{chatbot_type}_based_chatbot'))


def get_response_helper(user_query, data, clustering_model, vectors, tfidf_vectorizer, num_sentences=6, max_response_length=150):
    user_cluster = get_intent(user_query, clustering_model, tfidf_vectorizer)
    cluster_indices = np.where(clustering_model.labels_ == user_cluster)[0]
    cluster_responses = [data[i] for i in cluster_indices]

    # Handling Out-of-Cluster Queries - Fallback Mechanism
    if not cluster_responses:
        return get_fallback_response()

    # Context Matching
    sorted_responses = context_matching(user_query, cluster_responses)
    selected_responses = sorted_responses[:min(num_sentences, len(sorted_responses))]

    bot_response = " ".join(selected_responses)

    # Limit the response length
    if len(bot_response) > max_response_length:
        bot_response = " ".join(selected_responses[:num_sentences - 1])  # Use one less sentence if it exceeds the limit

    # Handling Ambiguous Queries
    if not bot_response:
        return get_fallback_response()
    return bot_response

# Calculate and display the average response relevance
def calculate_average_response_relevance(conversation):
    if not conversation:
        return 0  # No conversation, so relevance is 0
    
    relevance_scores = []

    # Calculate relevance for bot responses based on user questions
    for i in range(1, len(conversation), 2):
        user_question = conversation[i - 1]["text"]
        bot_response = conversation[i]["text"]
        
        similarity_score = calculate_similarity(user_question, bot_response)
        relevance_scores.append(similarity_score)

    if not relevance_scores:
        return 0  # No bot responses to evaluate relevance

    avg_relevance = sum(relevance_scores) / len(relevance_scores)
    return avg_relevance

@app.route('/llm_based_chatbot', methods=['GET', 'POST'])
def llm_based_chatbot():
    # Initialize or retrieve the conversation history from the session
    conversation = session.get('llm_conversation', [])
    
    if request.method == 'POST':
        user_question = request.form['user_question']
        output_preference = request.form['output_preference']

        # Append the user question to the conversation
        conversation.append({"type": "user", "text": user_question})

        # Get the bot response using the LLM chatbot
        generated_response = run_llm_chatbot(user_question, output_preference)

        # Append the bot response to the conversation
        conversation.append({"type": "bot", "text": generated_response})

        # Save the updated conversation history back to the session
        session['llm_conversation'] = conversation

        if output_preference == 'voice':
            speak(generated_response)

        avg_response_relevance = calculate_average_response_relevance(session.get('llm_conversation', []))
        print(f"Avg Response Relevance: {avg_response_relevance}")

    return render_template('llm_based_chatbot.html', conversation=session.get('llm_conversation', []))


@app.route('/')
def home():
    return render_template('/index.html')

if __name__ == "__main__":
    app.run(debug=True)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import homogeneity_score
from sklearn.metrics import silhouette_score
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

# Secret key
secret_key = secrets.token_hex(16)

# Setting the secret key
app.secret_key = secret_key

# Initialize the spaCy model with pre-trained embeddings
nlp = spacy.load("en_core_web_lg")

# Function to load data from transcript files
def load_data(folder_path):
    data = []
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data.append(line.strip())
    return data

# Function to preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def vectorize_data_with_embeddings(data):
    nlp = spacy.load("en_core_web_lg")
    vectors = [nlp(text).vector for text in data]
    return np.array(vectors), None

def agglomerative_cluster_data(vectors, num_clusters=5):
    clustering_model = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    clusters = clustering_model.fit_predict(vectors)
    return clusters, clustering_model

def calculate_cluster_homogeneity(clustering_model, vectors):
    labels_true = clustering_model.labels_
    if issparse(vectors):
        vectors = vectors.toarray()
    labels_pred = clustering_model.fit_predict(vectors)
    homogeneity = homogeneity_score(labels_true, labels_pred)
    return homogeneity

# Function to get fallback responses for out-of-cluster queries
def get_fallback_response():
    fallback_responses = [
        "I'm sorry, but I don't have information on that topic. Please try asking something else.",
        "I'm not sure I understand. Could you please rephrase your question?",
        "I'm still learning, and I don't have an answer for that. Can you try a different question?",
    ]
    return random.choice(fallback_responses)

def get_intent(user_query, clustering_model, vectorizer=None):
    if vectorizer:
        user_query_vector = vectorizer.transform([user_query])
    else:
        user_query_vector = np.array([nlp(user_query).vector])
    # Get the cluster labels for the existing vectors
    cluster_labels = clustering_model.labels_
    # Convert user_query_vector to a dense array if needed
    if vectorizer:
        user_query_vector = user_query_vector.toarray()
    # Append the user query vector to the existing data
    vectors_extended = np.concatenate([vectors, user_query_vector], axis=0)  # Concatenate along axis 0
    user_cluster = clustering_model.fit_predict(vectors_extended)[-1]  # Last element is the user query
    # Restore the original cluster labels
    clustering_model.labels_ = cluster_labels
    return user_cluster

def context_matching(user_query, responses):
    tfidf_vectorizer = TfidfVectorizer()
    response_vectors = tfidf_vectorizer.fit_transform(responses)
    user_query_vector = tfidf_vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(user_query_vector, response_vectors)[0]
    sorted_indices = np.argsort(similarity_scores)[::-1]
    sorted_responses = [responses[i] for i in sorted_indices]
    return sorted_responses

def calculate_silhouette_score(vectors, clustering_model):
    labels = clustering_model.labels_
    silhouette = silhouette_score(vectors, labels, metric='euclidean')
    return silhouette

# Function to generate speech output from text
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech (words per minute)
    engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
    engine.say("I will now start answering your question")
    engine.say(text)
    engine.runAndWait()

def vectorize_data_with_tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    vectors = tfidf_vectorizer.fit_transform(data)
    return vectors.toarray(), tfidf_vectorizer 

# Function to run the LLM-based chatbot
def run_llm_chatbot(question, output_preference):
    model_name = "static/distilgpt2_finetuned_model"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    model.eval()
    input_ids = tokenizer.encode(question, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_response

# Load and preprocess data
data_folder = "static/COMP3074transcripts"
data = load_data(data_folder)
preprocessed_data = [preprocess_text(item) for item in data]

# Vectorize data with TF-IDF
vectors, tfidf_vectorizer = vectorize_data_with_tfidf(preprocessed_data)

# Agglomerative Clustering
num_clusters = 5
clusters, clustering_model = agglomerative_cluster_data(vectors, num_clusters)

# Function to get or initialize the conversation history in the session
def get_or_init_conversation():
    if 'conversation' not in session:
        session['conversation'] = []  # Initialize an empty list for the conversation
    return session['conversation']

@app.route('/retrieval_based_chatbot', methods=['GET', 'POST'])
def retrieval_based_chatbot():
    # Initialize or retrieve the conversation history from the session
    conversation = session.get('conversation', [])
    
    if request.method == 'POST':
        # Get the user query and output preference from the form data
        user_query = request.form.get('user_query', '')
        output_preference = request.form.get('output_preference', 'text')

        # Pass the vectors and vectorizer as arguments to the get_response_helper function
        response = get_response_helper(user_query, data, clustering_model, vectors, tfidf_vectorizer)

        # Update the conversation history with the user query and chatbot response
        conversation.append({'role': 'user', 'content': user_query})
        conversation.append({'role': 'chatbot', 'content': response})

        if output_preference == 'voice':
            # Call the speak function to generate speech output
            speak(response)

        # Save the updated conversation history back to the session
        session['conversation'] = conversation

        if output_preference == 'text':
            return render_template('retrieval_based_chatbot.html', conversation=conversation)
    
    # If it's a GET request or the output_preference is 'voice', render the template directly
    return render_template('retrieval_based_chatbot.html', conversation=conversation)

@app.route('/end_conversation', methods=['POST'])
def end_conversation():
    chatbot_type = request.form.get('chatbot_type', 'retrieval')  # Default to "retrieval" chatbot
    if chatbot_type == 'retrieval':
        # Clear the retrieval-based chatbot's conversation history from the session
        session.pop('conversation', None)
    elif chatbot_type == 'llm':
        # Clear the LLM-based chatbot's conversation history from the session
        session.pop('llm_conversation', None)
    return redirect(url_for(f'{chatbot_type}_based_chatbot'))


def get_response_helper(user_query, data, clustering_model, vectors, tfidf_vectorizer, num_sentences=2, max_response_length=150):
    user_cluster = get_intent(user_query, clustering_model, tfidf_vectorizer)
    cluster_indices = np.where(clustering_model.labels_ == user_cluster)[0]
    cluster_responses = [data[i] for i in cluster_indices]

    # Handling Out-of-Cluster Queries - Fallback Mechanism
    if not cluster_responses:
        return get_fallback_response()

    # Context Matching
    sorted_responses = context_matching(user_query, cluster_responses)
    selected_responses = sorted_responses[:min(num_sentences, len(sorted_responses))]

    bot_response = " ".join(selected_responses)

    # Limit the response length
    if len(bot_response) > max_response_length:
        bot_response = " ".join(selected_responses[:num_sentences - 1])  # Use one less sentence if it exceeds the limit

    # Handling Ambiguous Queries
    if not bot_response:
        return get_fallback_response()
    return bot_response

@app.route('/llm_based_chatbot', methods=['GET', 'POST'])
def llm_based_chatbot():
    # Initialize or retrieve the conversation history from the session
    conversation = session.get('llm_conversation', [])
    
    if request.method == 'POST':
        user_question = request.form['user_question']
        output_preference = request.form['output_preference']

        # Append the user question to the conversation
        conversation.append({"type": "user", "text": user_question})

        # Get the bot response using the LLM chatbot
        generated_response = run_llm_chatbot(user_question, output_preference)

        # Append the bot response to the conversation
        conversation.append({"type": "bot", "text": generated_response})

        # Save the updated conversation history back to the session
        session['llm_conversation'] = conversation

        if output_preference == 'voice':
            speak(generated_response)

    return render_template('llm_based_chatbot.html', conversation=session.get('llm_conversation', []))

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
