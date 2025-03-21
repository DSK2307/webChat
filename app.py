import streamlit as st
import os
import tempfile
import requests
from bs4 import BeautifulSoup
import base64
from io import BytesIO
from PIL import Image
import re
from urllib.parse import urljoin, urlparse
import uuid
import imghdr

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# Set your API key directly in the code
# Replace "YOUR_API_KEY_HERE" with your actual API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBQKBh4BS0aq30-LZveeeGAJWDdQZq07Ok"

# Create a persistent directory for Chroma
PERSIST_DIRECTORY = os.path.join(tempfile.gettempdir(), "chroma_db")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

def get_vectorstore_from_url(url):
    try:
        # Get the text in document form
        loader = WebBaseLoader(url)
        document = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        
        # Create a vectorstore from the chunks using embeddings
        vector_store = Chroma.from_documents(
            documents=document_chunks, 
            embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
            persist_directory=PERSIST_DIRECTORY
        )
        vector_store.persist()
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise

def get_context_retriever_chain(vector_store):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        retriever = vector_store.as_retriever()
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        return retriever_chain
    except Exception as e:
        st.error(f"Error creating retriever chain: {str(e)}")
        raise
    
def get_conversational_rag_chain(retriever_chain): 
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever_chain, stuff_documents_chain)
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        raise

def get_response(user_input):
    try:
        # Check if the user is asking for images
        if re.search(r'(show|get|extract|display|download).*?(image|picture|photo|img)', user_input.lower()):
            if "website_url" in st.session_state and st.session_state.website_url:
                return f"[IMAGE_REQUEST]{st.session_state.website_url}"
            else:
                return "Please enter a valid website URL first to extract images."
        
        retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
        return response['answer']
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def is_valid_image(img_data):
    """Check if the data is a valid image."""
    try:
        img = Image.open(BytesIO(img_data))
        img.verify()
        return True
    except Exception:
        try:
            img_type = imghdr.what(None, h=img_data)
            return img_type is not None
        except Exception:
            return False

def extract_images_from_url(url):
    """Extract all images from a webpage and return their URLs and data."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        image_tags = soup.find_all('img')
        
        images = []
        for img in image_tags:
            img_url = img.get('src')
            if not img_url:
                continue
            if img_url.startswith('data:'):
                continue
            if not bool(urlparse(img_url).netloc):
                img_url = urljoin(url, img_url)
            alt_text = img.get('alt', f"Image {len(images) + 1}")
            if 'icon' in img_url.lower() or 'logo' in img_url.lower():
                if not alt_text or alt_text == f"Image {len(images) + 1}":
                    continue
            try:
                img_response = requests.get(img_url, headers=headers, timeout=5)
                img_response.raise_for_status()
                if is_valid_image(img_response.content):
                    images.append({
                        'url': img_url,
                        'alt_text': alt_text,
                        'data': img_response.content,
                        'id': str(uuid.uuid4())
                    })
            except Exception:
                continue
        
        return images
    except Exception as e:
        st.error(f"Error extracting images: {str(e)}")
        return []

def get_image_download_link(img_data, filename, text, mime="image/png"):
    """Generate a download link for an image."""
    b64 = base64.b64encode(img_data).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">📥 {text}</a>'
    return href

# App config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    if website_url:
        st.session_state.website_url = website_url
    
    if st.button("Reset Vector Store"):
        if "vector_store" in st.session_state:
            del st.session_state.vector_store
        st.success("Vector store reset successfully!")
    
    if st.button("Extract Images") and "website_url" in st.session_state:
        with st.spinner("Extracting images..."):
            st.session_state.extracted_images = extract_images_from_url(st.session_state.website_url)
        if st.session_state.extracted_images:
            st.success(f"Extracted {len(st.session_state.extracted_images)} images!")
        else:
            st.warning("No images found on this website.")

# Main app
if "website_url" not in st.session_state or st.session_state.website_url == "":
    st.info("Please enter a website URL in the sidebar.")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am an AI assistant. How can I help you?")
        ]
        
    if "vector_store" not in st.session_state:
        try:
            with st.spinner("Loading website content and creating embeddings..."):
                st.session_state.vector_store = get_vectorstore_from_url(st.session_state.website_url)
                st.success("Website content loaded successfully!")
        except Exception as e:
            st.error(f"Error loading website: {str(e)}")
    
    user_query = st.chat_input("Type your message here...")
    if user_query and "vector_store" in st.session_state:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.spinner("Generating response..."):
            try:
                response = get_response(user_query)
                if response.startswith("[IMAGE_REQUEST]"):
                    url = response.replace("[IMAGE_REQUEST]", "").strip()
                    st.session_state.extracted_images = extract_images_from_url(url)
                    if st.session_state.extracted_images:
                        image_message = f"I found {len(st.session_state.extracted_images)} images on this website. Here they are:"
                        st.session_state.chat_history.append(AIMessage(content=image_message))
                        st.session_state.show_images = True
                    else:
                        st.session_state.chat_history.append(AIMessage(content="I couldn't find any images on this website."))
                else:
                    st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
    
    if st.session_state.get("show_images") and st.session_state.get("extracted_images"):
        with st.expander("Extracted Images", expanded=True):
            cols = st.columns(3)
            valid_images = 0
            for img_data in st.session_state.extracted_images:
                try:
                    img_bytes = BytesIO(img_data['data'])
                    img_bytes.seek(0)
                    image = Image.open(img_bytes)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    img_format = image.format if image.format else "PNG"
                    mime = f"image/{img_format.lower()}"
                    
                    col = cols[valid_images % 3]
                    col.image(image, caption=img_data['alt_text'], use_container_width=True)
                    
                    filename = f"image_{img_data['id']}.{img_format.lower()}"
                    download_link = get_image_download_link(img_data['data'], filename, "Download", mime=mime)
                    col.markdown(download_link, unsafe_allow_html=True)
                    valid_images += 1
                except Exception:
                    continue
            
            if valid_images == 0:
                st.warning("None of the extracted images could be displayed.")
            
        st.session_state.show_images = False