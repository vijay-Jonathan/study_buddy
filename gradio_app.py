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
    """Extract text from PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
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
        # Extract text from PDFs
        raw_text = get_pdf_text(files)
        if raw_text.strip():
            # Create text chunks
            text_chunks = get_text_chunks(raw_text)
            # Create vector store
            vector_store = get_vector_store(text_chunks)
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
        return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": ai_response}]
    
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
        return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": ai_response}]
        
    except Exception as e:
        error_message = f"Sorry, I encountered an error: {str(e)}"
        return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_message}]

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
    
    # Event handlers
    process_btn.click(
        process_documents,
        inputs=[file_input],
        outputs=[status_display, msg]
    )
    
    msg.submit(
        chat_response,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    submit_btn.click(
        chat_response,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
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
