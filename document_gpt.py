import streamlit as st
from sentence_transformers import SentenceTransformer, util
import anthropic
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access API key from environment variable
ANTHROPIC_API_KEY = st.secrets['ANTHROPIC_API_KEY']
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Load a smaller, pre-trained sentence transformer model for document embedding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load and process uploaded documents
def load_documents(files):
    documents = []
    for file in files:
        # Read the content of the uploaded file
        content = file.read().decode('utf-8')
        documents.append(content)
    return documents

# Function to retrieve relevant documents
def retrieve_documents(query, document_embeddings, documents, top_k=2):
    # Encode the query
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    
    # Perform semantic search
    hits = util.semantic_search(query_embedding, document_embeddings, top_k=top_k)
    hits = hits[0]  # Get the list of hits for the query

    # Return the top retrieved documents
    return [documents[hit['corpus_id']] for hit in hits]

# Function to generate a response using Claude AI
def generate_response(persona, retrieved_docs, user_query):
    client = anthropic.Anthropic()
    combined_input = f"{' '.join(retrieved_docs)}\n\nUser's Query: {user_query}"
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.2,
        system=persona,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": combined_input
                    }
                ]
            }
        ]
    )

    # Extract and return the text content from the message
    response_text = [block.text for block in message.content]
    return ''.join(response_text).strip()

# Streamlit UI
st.title("RAG Application with Claude AI")
st.write("Upload your documents, ask a question, and I'll retrieve relevant information and generate a response.")

# File upload component
uploaded_files = st.file_uploader("Upload text files", type=['txt'], accept_multiple_files=True)

# Initialize variables
documents = []
document_embeddings = None

if uploaded_files:
    # Load and encode uploaded documents
    documents = load_documents(uploaded_files)
    document_embeddings = embedding_model.encode(documents, convert_to_tensor=True)
    st.success(f"Successfully loaded and processed {len(documents)} documents.")

with st.form("query_form"):
    describe_persona = st.text_input("Describe the persona for the model", placeholder="You are a world-class assistant")
    user_query = st.text_input("Enter your question", placeholder="Ask me anything...")
    submitted = st.form_submit_button("Send")

    if submitted and user_query.strip() != "" and documents:
        with st.spinner('Retrieving and Generating...'):
            retrieved_docs = retrieve_documents(user_query, document_embeddings, documents)
            model_output = generate_response(describe_persona, retrieved_docs, user_query)
            st.markdown(f"**You:** {user_query}") 
            st.markdown(f"**Claude AI:** {model_output}")
    elif submitted and not documents:
        st.warning("Please upload documents before asking a question.")

st.write("Built with Streamlit, Claude AI, and Hugging Face")
