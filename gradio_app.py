import gradio as gr
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# Global variables for processed data
processed_documents = False
vector_store = None

def get_pdf_text(pdf_docs):
    """Extract text from PDFs with document names."""
    text_chunks_with_metadata = []
    
    for pdf in pdf_docs:
        # Get document name
        doc_name = getattr(pdf, 'name', 'Unknown Document')
        if not doc_name or doc_name == 'Unknown Document':
            doc_name = f"document_{len(text_chunks_with_metadata) + 1}.pdf"
        
        pdf_reader = PdfReader(pdf)
        
        # Extract text from each page with metadata
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                # Create metadata for this chunk
                metadata = {
                    'source': doc_name,
                    'page': page_num + 1,
                    'document': doc_name
                }
                text_chunks_with_metadata.append({
                    'text': page_text,
                    'metadata': metadata
                })
    
    return text_chunks_with_metadata

def get_text_chunks(text_chunks_with_metadata):
    """Split text into manageable chunks while preserving metadata."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    
    all_chunks = []
    
    for doc_data in text_chunks_with_metadata:
        text = doc_data['text']
        base_metadata = doc_data['metadata']
        
        # Split the text
        chunks = text_splitter.split_text(text)
        
        # Create chunk with metadata for each text chunk
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata['chunk'] = i + 1
            all_chunks.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
    
    return all_chunks

def get_vector_store(text_chunks_with_metadata):
    """Create and save vector store with metadata."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Extract texts and create documents with metadata
    texts = [chunk['text'] for chunk in text_chunks_with_metadata]
    metadatas = [chunk['metadata'] for chunk in text_chunks_with_metadata]
    
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    """Create the conversational chain."""
    prompt_template = """
    You are a helpful AI assistant for studying documents. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer

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
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "conversation_history"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def format_conversation_history(chat_history, max_turns=5):
    """Format recent conversation history for context."""
    if not chat_history:
        return "No previous conversation."
    
    # Get only the last few turns to avoid context overflow
    recent_history = chat_history[-max_turns*2:] if len(chat_history) > max_turns*2 else chat_history
    
    formatted_history = "Recent conversation:\n"
    for i, message in enumerate(recent_history):
        if isinstance(message, dict):
            if message.get("role") == "user":
                formatted_history += f"User Question {i//2 + 1}: {message.get('content', '')}\n"
            elif message.get("role") == "assistant":
                formatted_history += f"Assistant Answer {i//2 + 1}: {message.get('content', '')}\n"
        else:
            # Handle old format (list of lists) for backward compatibility
            if len(message) >= 2:
                formatted_history += f"User Question {i//2 + 1}: {message[0]}\n"
                formatted_history += f"Assistant Answer {i//2 + 1}: {message[1]}\n"
    
    return formatted_history.strip()

def process_documents(files):
    """Process uploaded PDF documents."""
    global processed_documents, vector_store
    
    if not files:
        return "Please upload PDF files first.", gr.update(interactive=False)
    
    try:
        # Extract text from PDFs with metadata
        text_chunks_with_metadata = get_pdf_text(files)
        if text_chunks_with_metadata:
            # Create text chunks with preserved metadata
            chunks_with_metadata = get_text_chunks(text_chunks_with_metadata)
            # Create vector store with metadata
            vector_store = get_vector_store(chunks_with_metadata)
            processed_documents = True
            return f"Successfully processed {len(files)} document(s)!", gr.update(interactive=True)
        else:
            return "No text found in the uploaded PDFs. Please try different files.", gr.update(interactive=False)
    except Exception as e:
        return f"Error processing documents: {str(e)}", gr.update(interactive=False)

def chat_response(message, history):
    """Generate AI response for user message."""
    global processed_documents, vector_store
    
    if not processed_documents:
        ai_response = "Please upload and process PDF documents first."
        return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": ai_response}], "No sources available"
    
    try:
        # Format conversation history for context
        conversation_history = format_conversation_history(history)
        
        # Load vector store and search for relevant documents
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(message)
        
        # Get conversational chain and generate response
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": message, "conversation_history": conversation_history},
            return_only_outputs=True
        )
        
        ai_response = response["output_text"]
        
        # Format sources for display
        sources_text = format_sources_for_display(docs)
        
        return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": ai_response}], sources_text
        
    except Exception as e:
        error_message = f"Sorry, I encountered an error: {str(e)}"
        return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_message}], "Error occurred while retrieving sources"

def format_sources_for_display(docs):
    """Format the source documents for display in the sources panel."""
    if not docs:
        return "No sources found"
    
    sources_text = "**Referenced Sources:**\n\n"
    
    for i, doc in enumerate(docs, 1):
        # Get metadata if available
        if hasattr(doc, 'metadata') and doc.metadata:
            # Get document name from metadata
            doc_name = doc.metadata.get('document', 'Unknown Document')
            if not doc_name or doc_name == 'Unknown Document':
                doc_name = doc.metadata.get('source', 'Unknown Document')
            
            # Extract actual filename from path
            if '/' in doc_name:
                doc_name = doc_name.split('/')[-1]  # Get last part of path
            
            # Get page number
            page_num = doc.metadata.get('page', 'Unknown')
            if page_num != 'Unknown':
                page_info = f" (Page {page_num})"
            else:
                page_info = ""
            
            sources_text += f"**{i}.** {doc_name}{page_info}\n"
        else:
            sources_text += f"**{i}.** Unknown Document\n"
    
    return sources_text

def clear_chat():
    """Clear the chat history."""
    return [], []

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("""
    # Student Buddy - AI Study Assistant
    Upload your study materials and chat with them using AI!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Document Manager")
            
            # File upload section
            file_input = gr.File(
                label="Upload PDF Files", 
                file_count="multiple",
                file_types=[".pdf"],
                height=200
            )
            
            # Process button
            process_btn = gr.Button("Process Documents", size="lg")
            
            # Status display
            status_display = gr.Textbox(
                label="Status", 
                interactive=False,
                placeholder="Upload PDFs and click Process Documents"
            )
            
            # Clear chat button
            clear_btn = gr.Button("Clear Chat")
            
            gr.Markdown("---")
            gr.Markdown("### Instructions")
            gr.Markdown("""
            1. Upload your PDF study materials
            2. Click 'Process Documents' 
            3. Start asking questions!
            4. AI remembers your conversation
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("## Study Chat")
            
            # Chat interface
            chatbot = gr.Chatbot(
                label="Chat with your materials",
                height=600
            )
            
            # Message input
            with gr.Row():
                msg = gr.Textbox(
                    label="Ask a question about your documents",
                    placeholder="Type your question here...",
                    scale=4
                )
                submit_btn = gr.Button("Send", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("## Sources")
            
            # Sources display
            sources_display = gr.Markdown(
                value="**Sources will appear here**\n\nAsk a question to see referenced documents.",
                label="Referenced Sources"
            )
    
    # Event handlers
    process_btn.click(
        process_documents,
        inputs=[file_input],
        outputs=[status_display, msg]
    )
    
    msg.submit(
        chat_response,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, sources_display]
    )
    
    submit_btn.click(
        chat_response,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, sources_display]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates public link
        debug=True,  # Enable debug mode
        show_error=True  # Show error messages
    )
