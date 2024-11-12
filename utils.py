import streamlit as st
import pandas as pd
import json
import base64
from cryptography.fernet import Fernet
import re
import sqlite3
import PyPDF2
import logging
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_custom_style():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f7f8fa;
        color: #333;
        font-size: 16px;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #0073e6;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #005bb5;
    }
    .sidebar .sidebar-content {
        background-color: #f7f8fa;
        color: #333;
    }
    .st-expander {
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def encrypt_api_key(api_key):
    try:
        key = Fernet.generate_key()
        f = Fernet(key)
        return f.encrypt(api_key.encode()), key
    except Exception as e:
        logger.error(f"Error encrypting API key: {str(e)}")
        raise

def decrypt_api_key(encrypted_key, key):
    try:
        f = Fernet(key)
        return f.decrypt(encrypted_key).decode()
    except Exception as e:
        logger.error(f"Error decrypting API key: {str(e)}")
        raise

def get_conversations_from_db():
    try:
        conn = sqlite3.connect('responses.db')
        c = conn.cursor()
        c.execute("SELECT conversation_id, title, conversation_history FROM conversations ORDER BY timestamp DESC")
        result = c.fetchall()
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error retrieving conversations from database: {str(e)}")
        raise

def search_conversations(search_term):
    try:
        conn = sqlite3.connect('responses.db')
        c = conn.cursor()
        c.execute(
            "SELECT conversation_id, title, conversation_history FROM conversations WHERE title LIKE ? OR conversation_history LIKE ?",
            (f"%{search_term}%", f"%{search_term}%"))
        result = c.fetchall()
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error searching conversations: {str(e)}")
        raise

def get_all_documents_from_db():
    try:
        conn = sqlite3.connect('responses.db')
        c = conn.cursor()
        c.execute("SELECT filename, content FROM documents ORDER BY timestamp DESC")
        result = c.fetchall()
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error retrieving documents from database: {str(e)}")
        raise

def generate_conversation_summary(conversation_history):
    if not conversation_history:
        return "No messages yet"
    last_message = conversation_history[-1]['content']
    return last_message[:100] + "..." if len(last_message) > 100 else last_message

def extract_and_visualize_table(markdown_text):
    table_pattern = r'(\|[^|\n]+\|[^|\n]+\n\|[-:| ]+\n(?:\|[^|\n]+\n)+)'
    tables = re.findall(table_pattern, markdown_text, re.MULTILINE)
    if not tables:
        return None

    table_data = tables[0]
    df = pd.read_csv(pd.compat.StringIO(table_data), sep='|', skipinitialspace=True)
    df = df.dropna(axis=1, how='all').iloc[:, 1:-1]
    df.columns = df.columns.str.strip()

    st.dataframe(df)

    if df.shape[1] >= 2:
        chart = px.line(df, x=df.columns[0], y=df.columns[1])
        st.plotly_chart(chart)

    return df

def export_conversation(conversation_history):
    return base64.b64encode(json.dumps(conversation_history).encode()).decode()

def import_conversation(encoded_conversation):
    return json.loads(base64.b64decode(encoded_conversation).decode())

def save_document_to_db(filename, content):
    try:
        conn = sqlite3.connect('responses.db')
        c = conn.cursor()
        
        # Save document to SQLite
        c.execute("INSERT INTO documents (filename, content) VALUES (?, ?)", (filename, content))
        
        conn.commit()
        conn.close()
        logger.info(f"Document saved to database: {filename}")
    except Exception as e:
        logger.error(f"Error saving document to database: {str(e)}")
        raise

def get_documents_from_db():
    try:
        conn = sqlite3.connect('responses.db')
        c = conn.cursor()
        c.execute("SELECT id, filename FROM documents ORDER BY timestamp DESC")
        result = c.fetchall()
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error retrieving documents from database: {str(e)}")
        raise

def get_document_content(doc_id):
    try:
        conn = sqlite3.connect('responses.db')
        c = conn.cursor()
        c.execute("SELECT content FROM documents WHERE id = ?", (doc_id,))
        result = c.fetchone()[0]
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error retrieving document content: {str(e)}")
        raise

def search_similar_documents(query, k=5):
    try:
        conn = sqlite3.connect('responses.db')
        c = conn.cursor()
        
        # Simple text-based search using LIKE
        c.execute("""
            SELECT filename, content 
            FROM documents 
            WHERE content LIKE ? 
            LIMIT ?
        """, (f"%{query}%", k))
        
        results = c.fetchall()
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def save_user_feedback(conversation_id, rating, comment):
    try:
        conn = sqlite3.connect('responses.db')
        c = conn.cursor()
        c.execute("INSERT INTO user_feedback (conversation_id, rating, comment) VALUES (?, ?, ?)",
                  (conversation_id, rating, comment))
        conn.commit()
        conn.close()
        logger.info(f"User feedback saved for conversation: {conversation_id}")
    except Exception as e:
        logger.error(f"Error saving user feedback: {str(e)}")
        raise

def get_user_feedback():
    try:
        conn = sqlite3.connect('responses.db')
        c = conn.cursor()
        c.execute("SELECT conversation_id, rating, comment, timestamp FROM user_feedback ORDER BY timestamp DESC")
        result = c.fetchall()
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error retrieving user feedback: {str(e)}")
        raise

def generate_conversation_title(question, max_length=50):
    # Remove special characters and extra spaces
    clean_question = re.sub(r'[^\w\s]', '', question).strip()
    
    # Split the question into words
    words = clean_question.split()
    
    # If the question is already short enough, return it as is
    if len(clean_question) <= max_length:
        return clean_question
    
    # Otherwise, truncate the question and add ellipsis
    title = ""
    for word in words:
        if len(title) + len(word) + 1 > max_length - 3:  # -3 for the ellipsis
            break
        title += word + " "
    
    return title.strip() + "..."

def save_conversation_to_db(conversation_id, question, conversation_history):
    try:
        conn = sqlite3.connect('responses.db')
        c = conn.cursor()
        title = generate_conversation_title(question)
        c.execute("INSERT OR REPLACE INTO conversations (conversation_id, title, conversation_history) VALUES (?, ?, ?)",
                  (conversation_id, title, json.dumps(conversation_history)))
        conn.commit()
        conn.close()
        logger.info(f"Conversation saved to database: {conversation_id}")
    except Exception as e:
        logger.error(f"Error saving conversation to database: {str(e)}")
        raise

# Initialize the database schema
def init_db():
    conn = sqlite3.connect('responses.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, filename TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Call this function when the application starts
init_db()