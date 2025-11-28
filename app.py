import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import openai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    


def get_conversational_chain():

    prompt_template = """
    You are a helpful AI assistant for studying documents. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n


    IMPORTANT: Handle conversational follow-ups naturally! If the user asks about previous questions, conversation, or follow-ups, 
    respond conversationally using the conversation history provided.
    
    Conversation History:
    {conversation_history}
    
    Context:
    {context}
    
    Question: {question}
    
    Instructions:
    1. If the question is about the conversation (like "what did I ask before?"), use the conversation history
    2. If the question is about document content, use the context
    3. If the question combines both, use both sources
    4. If information is not available in either source, say it's not available
    
    Answer:
    """

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question", "conversation_history"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config(page_title="PDF ChatBot", layout="centered")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed" not in st.session_state:
        st.session_state.processed = False
    
    # Custom CSS for simple chat styling
    st.markdown("""
    <style>
        .chat-container {
            max-width: 700px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 12px 16px;
            margin: 10px 0;
            border-radius: 8px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background-color: #f1f1f1;
            color: #333;
            margin-right: auto;
        }
        .chat-header {
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    

    
    # Sidebar for document upload
    with st.sidebar:
        st.title("Document Manager")
        
        # File upload section
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Choose PDF files", 
            accept_multiple_files=True,
            type="pdf"
        )
        
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text.strip():
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.session_state.processed = True
                            st.success(f"Successfully processed {len(pdf_docs)} document(s)!")
                        else:
                            st.error("No text found in the uploaded PDFs. Please try different files.")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please upload PDF files first.")
        
        # Document status
        st.subheader("Status")
        if st.session_state.processed:
            st.success("Documents ready for chat!")
        else:
            st.info("Please upload and process documents to start chatting.")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    if st.session_state.processed:
        # Input form at the bottom
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_question = st.text_input(
                    "Type your message...",
                    placeholder="Ask anything about your documents",
                    key="user_input"
                )
            with col2:
                submit_button = st.form_submit_button("Send")
            
            if submit_button and user_question:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Get AI response
                with st.spinner("Thinking..."):
                    try:
                        # Format conversation history for context
                        conversation_history = format_conversation_history(st.session_state.chat_history)
                        response = get_conversational_response(user_question, conversation_history)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": f"Sorry, I encountered an error: {str(e)}"
                        })
                
                st.rerun()
    else:
        st.info("Please upload and process PDF documents in the sidebar to start chatting.")


def format_conversation_history(chat_history, max_turns=5):
    """Format recent conversation history for context."""
    if not chat_history:
        return "No previous conversation."
    
    # Get only the last few turns to avoid context overflow
    recent_history = chat_history[-max_turns*2:] if len(chat_history) > max_turns*2 else chat_history
    
    formatted_history = "Recent conversation:\n"
    for i, message in enumerate(recent_history):
        if message["role"] == "user":
            formatted_history += f"User Question {i//2 + 1}: {message['content']}\n"
        else:
            formatted_history += f"Assistant Answer {i//2 + 1}: {message['content']}\n"
    
    return formatted_history.strip()


def get_conversational_response(user_question, conversation_history=""):
    """Get response from the conversational chain with memory."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    # Include conversation history for context
    context_with_history = ""
    if conversation_history:
        context_with_history = f"Previous conversation:\n{conversation_history}\n\n"
    
    response = chain(
        {"input_documents": docs, "question": user_question, "conversation_history": context_with_history},
        return_only_outputs=True
    )
    
    return response["output_text"]



if __name__ == "__main__":
    main()