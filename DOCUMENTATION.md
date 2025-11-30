# Student Buddy - AI Study Assistant Documentation

## Overview
Student Buddy is an AI-powered study assistant that allows students to upload PDF documents (textbooks, notes, assignments) and chat with them using advanced RAG (Retrieval-Augmented Generation) technology.

## Tech Stack
- **Frontend**: Gradio (Web Interface)
- **Backend**: Python
- **AI Models**: 
  - Hugging Face (Embeddings: sentence-transformers/all-mpnet-base-v2)
  - OpenAI (LLM: gpt-3.5-turbo)
- **Vector Database**: FAISS
- **Framework**: LangChain
- **Document Processing**: PyPDF2

## Architecture
```
User Uploads PDF → Text Extraction → Chunking → Embedding Generation → Vector Store (FAISS)
                                                                 ↓
User Question → Similarity Search → Context Retrieval → LLM Generation → Response
```

---

## Code Documentation: gradio_app.py

### Import Statements (Lines 1-13)

```python
import gradio as gr
```
**Purpose**: Import Gradio library for creating web interfaces
**Functionality**: Provides UI components like File upload, Chatbot, Textbox, Buttons

```python
import os
```
**Purpose**: Operating system interface
**Functionality**: Access environment variables for API keys

```python
from PyPDF2 import PdfReader
```
**Purpose**: PDF processing library
**Functionality**: Extract text content from PDF files

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```
**Purpose**: Text chunking from LangChain
**Functionality**: Split large documents into smaller, manageable chunks for processing

```python
from langchain_openai import ChatOpenAI
```
**Purpose**: OpenAI chat model integration
**Functionality**: Interface with OpenAI's GPT models for response generation

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
```
**Purpose**: Hugging Face embeddings model
**Functionality**: Convert text chunks into numerical vectors for similarity search

```python
import openai
```
**Purpose**: OpenAI Python client
**Functionality**: Direct API access to OpenAI services

```python
from langchain_community.vectorstores import FAISS
```
**Purpose**: Facebook AI Similarity Search
**Functionality**: Vector database for efficient similarity search and retrieval

```python
from langchain.chains.question_answering import load_qa_chain
```
**Purpose**: Question-answering chain from LangChain
**Functionality**: Orchestrate the RAG pipeline (retrieval + generation)

```python
from langchain.prompts import PromptTemplate
```
**Purpose**: Template management for prompts
**Functionality**: Structure and format prompts sent to the LLM

```python
from dotenv import load_dotenv
```
**Purpose**: Environment variable management
**Functionality**: Load API keys and configuration from .env file

### Configuration Setup (Lines 15-20)

```python
load_dotenv()
```
**Purpose**: Load environment variables from .env file
**Functionality**: Securely load API keys without hardcoding them

```python
os.getenv("OPENAI_API_KEY")
```
**Purpose**: Retrieve OpenAI API key from environment
**Functionality**: Get API key for OpenAI authentication

```python
openai.api_key = os.getenv("OPENAI_API_KEY")
```
**Purpose**: Set OpenAI API key for the client
**Functionality**: Configure OpenAI client with authentication

### Global Variables (Lines 22-25)

```python
# Global variables for processed data
processed_documents = False
```
**Purpose**: Track document processing status
**Functionality**: Boolean flag to indicate if documents have been processed

```python
vector_store = None
```
**Purpose**: Store the FAISS vector database
**Functionality**: Global reference to the vector store for document retrieval

---

## Core Functions

### PDF Text Extraction with Document Names (Lines 27-44)

```python
def get_pdf_text(pdf_docs):
    """Extract text from PDFs and capture document names."""
    text = ""
    document_names = []
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text
        
        if pdf_text:
            text += pdf_text
            # Store document name with page information
            doc_name = getattr(pdf, 'name', f'document_{len(document_names)}')
            document_names.append({
                'name': doc_name,
                'pages': len(pdf_reader.pages),
                'text_length': len(pdf_text)
            })
    
    return text, document_names
```

**Purpose**: Extract text from uploaded PDF files and capture metadata
**Parameters**: 
- `pdf_docs`: List of uploaded PDF file objects

**Process**:
1. Initialize empty text string and document names list
2. Iterate through each PDF file
3. Create PdfReader object for each PDF
4. Extract text from each page using `extract_text()`
5. Capture document metadata (name, pages, text length)
6. Concatenate all page text into single string
7. Return combined text and document metadata

**Returns**: Tuple of (text_string, document_metadata_list)

### Enhanced Text Chunking with Metadata (Lines 46-56)

```python
def get_text_chunks(text, document_names):
    """Split text into manageable chunks with document metadata."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    
    # Create chunks with metadata
    chunks_with_metadata = []
    for i, chunk in enumerate(chunks):
        chunks_with_metadata.append({
            'text': chunk,
            'chunk_id': i,
            'document': document_names[0]['name'] if document_names else 'Unknown'
        })
    
    return chunks_with_metadata
```

**Purpose**: Split large text into smaller, overlapping chunks with metadata
**Parameters**:
- `text`: String containing extracted PDF text
- `document_names`: List of document metadata

**Process**:
1. Create RecursiveCharacterTextSplitter with:
   - `chunk_size=10000`: Maximum characters per chunk
   - `chunk_overlap=1000`: Overlapping characters between chunks
2. Split text using the splitter
3. Create metadata-rich chunks with document information
4. Return list of chunk dictionaries

**Returns**: List of text chunks with metadata

### Vector Store Creation with Metadata (Lines 58-65)

```python
def get_vector_store(chunks_with_metadata):
    """Create and save vector store with metadata."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Extract text and metadata for FAISS
    texts = [chunk['text'] for chunk in chunks_with_metadata]
    metadatas = [{'document': chunk['document'], 'chunk_id': chunk['chunk_id']} for chunk in chunks_with_metadata]
    
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")
    return vector_store
```

**Purpose**: Create and persist vector database from text chunks with metadata
**Parameters**:
- `chunks_with_metadata`: List of text chunks with metadata

**Process**:
1. Initialize HuggingFace embeddings model
2. Extract text content and metadata separately
3. Create FAISS vector store with text, embeddings, and metadata
4. Save vector store locally to "faiss_index" directory
5. Return the created vector store

**Returns**: FAISS vector store object with embedded metadata

### Conversational Chain Setup (Lines 67-99)

```python
def get_conversational_chain():
```
**Purpose**: Create the RAG pipeline for question answering

**Prompt Template** (Lines 68-83):
```python
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
```

**Template Variables**:
- `{conversation_history}`: Previous chat messages for context
- `{context}`: Retrieved document chunks
- `{question}`: Current user question

**Model Configuration** (Lines 84-85):
```python
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
```
- Model: `gpt-3.5-turbo` - OpenAI's chat model
- Temperature: `0.3` - Low randomness for consistent, factual responses

**Chain Creation** (Lines 86-88):
```python
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "conversation_history"])
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
```
- Create PromptTemplate with specified variables
- Load "stuff" chain (puts all retrieved context into prompt)
- Return configured chain

**Returns**: Configured LangChain QA chain

### Conversation History Formatting (Lines 91-112)

```python
def format_conversation_history(chat_history, max_turns=5):
```
**Purpose**: Format chat history for inclusion in prompts
**Parameters**:
- `chat_history`: List of chat messages in Gradio format
- `max_turns`: Maximum number of conversation turns to include

**Process**:
1. Check if history is empty, return default message
2. Limit history to recent messages (last `max_turns * 2` messages)
3. Format each message as "User Question X: [content]" or "Assistant Answer X: [content]"
4. Handle both new format (dict with role/content) and old format (list)
5. Return formatted string

**Returns**: Formatted conversation history string

### Document Processing with Metadata (Lines 114-140)

```python
def process_documents(files):
```
**Purpose**: Process uploaded PDF files and create vector store with metadata
**Parameters**:
- `files`: List of uploaded files from Gradio file component

**Process**:
1. Check if files are uploaded, return error if not
2. Extract text and metadata using `get_pdf_text()`
3. Check if text was extracted successfully
4. Create chunks with metadata using `get_text_chunks()`
5. Generate vector store using `get_vector_store()`
6. Set global `processed_documents` flag to True
7. Return success message and enable chat input

**Error Handling**:
- No files uploaded: Return error message
- No text found in PDFs: Return error message
- Processing exceptions: Return error with details

**Returns**: Tuple of (status_message, input_component_update)

### Enhanced Chat Response with Sources (Lines 142-185)

```python
def chat_response(message, history):
```
**Purpose**: Generate AI responses to user questions with source attribution
**Parameters**:
- `message`: Current user question
- `history`: Conversation history from Gradio chatbot

**Process**:
1. Check if documents are processed, return prompt if not
2. Format conversation history using `format_conversation_history()`
3. Load FAISS vector store from disk
4. Perform similarity search to find relevant document chunks
5. Get conversational chain
6. Generate response using chain with:
   - Retrieved documents as context
   - User question
   - Conversation history
7. Extract sources from retrieved documents
8. Format sources for clean display
9. Return response, updated chat history, and formatted sources

**Error Handling**: Catch and format any exceptions gracefully

**Returns**: Tuple of (empty_string, updated_chat_history, sources_display)

### Sources Formatting (Lines 187-205)

```python
def format_sources_for_display(docs):
    """Format document sources for clean display."""
    if not docs:
        return "No sources found"
    
    sources_text = "**Referenced Sources:**\n\n"
    for i, doc in enumerate(docs, 1):
        if hasattr(doc, 'metadata') and doc.metadata:
            doc_name = doc.metadata.get('document', 'Unknown Document')
            # Extract filename from path if it's a full path
            if '/' in doc_name:
                doc_name = doc_name.split('/')[-1]
            
            page_num = doc.metadata.get('page', 'Unknown')
            page_info = f" (Page {page_num})" if page_num != 'Unknown' else ""
            
            sources_text += f"**{i}.** {doc_name}{page_info}\n"
        else:
            sources_text += f"**{i}.** Unknown Document\n"
    
    return sources_text
```

**Purpose**: Format source documents for clean, readable display
**Parameters**:
- `docs`: List of retrieved documents with metadata

**Process**:
1. Check if documents exist, return message if none
2. Iterate through each document
3. Extract document name and page from metadata
4. Clean up document names (remove paths)
5. Format as numbered list with document names and page numbers
6. Return formatted sources text

**Returns**: Formatted sources string for display

### Chat Clearing (Lines 207-209)

```python
def clear_chat():
    """Clear the chat history."""
    return [], []
```
**Purpose**: Reset the chat interface
**Returns**: Tuple of empty lists for chatbot and input components

---

## Gradio Interface with Enhanced UI (Lines 211-300)

### Interface Creation (Lines 211-213)

```python
with gr.Blocks() as demo:
```
**Purpose**: Create Gradio Blocks interface
**Functionality**: Container for all UI components

### Header (Lines 214-218)

```python
gr.Markdown("""
# Student Buddy - AI Study Assistant
Upload your study materials and chat with them using AI!
""")
```
**Purpose**: Display application title and description

### Layout Structure (Lines 220-224)

```python
with gr.Row():
    with gr.Column(scale=1):
        # Document Manager
    with gr.Column(scale=2):
        # Study Chat
    with gr.Column(scale=1):
        # Sources Panel
```
**Purpose**: Create responsive three-column layout
- Left column: Document management (scale=1)
- Center column: Chat interface (scale=2, larger)
- Right column: Sources display (scale=1)

### Document Management Panel (Lines 225-260)

#### File Upload (Lines 227-233)
```python
file_input = gr.File(
    label="Upload PDF Files", 
    file_count="multiple",
    file_types=[".pdf"],
    height=200
)
```
**Purpose**: Allow multiple PDF file uploads
**Options**:
- Multiple file selection
- PDF file type filter
- Custom height for better UX

#### Process Button (Lines 235-236)
```python
process_btn = gr.Button("Process Documents", size="lg")
```
**Purpose**: Trigger document processing
**Styling**: Large button for prominence

#### Status Display (Lines 238-242)
```python
status_display = gr.Textbox(
    label="Status", 
    interactive=False,
    placeholder="Upload PDFs and click Process Documents"
)
```
**Purpose**: Show processing status and feedback
**Features**: Non-interactive, placeholder text

#### Clear Button (Lines 244-245)
```python
clear_btn = gr.Button("Clear Chat")
```
**Purpose**: Clear chat history

#### Instructions (Lines 247-260)
```python
gr.Markdown("---")
gr.Markdown("### Instructions")
gr.Markdown("""
1. Upload your PDF study materials
2. Click 'Process Documents' 
3. Start asking questions!
4. AI remembers your conversation
""")
```
**Purpose**: Provide user guidance

### Chat Interface Panel (Lines 262-285)

#### Chatbot Component (Lines 264-267)
```python
chatbot = gr.Chatbot(
    label="Chat with your materials",
    height=600
)
```
**Purpose**: Display conversation history
**Features**: 
- Custom label
- Fixed height for better UX
- Message bubbles

#### Input Section (Lines 269-276)
```python
with gr.Row():
    msg = gr.Textbox(
        label="Ask a question about your documents",
        placeholder="Type your question here...",
        scale=4
    )
    submit_btn = gr.Button("Send", scale=1)
```
**Purpose**: User input interface
**Layout**: Row with textbox (4x scale) and button (1x scale)

### Sources Panel (Lines 287-295)

```python
gr.Markdown("## Sources")
sources_display = gr.Markdown(
    value="**Sources will appear here**\n\nAsk a question to see referenced documents.",
    label="Referenced Sources"
)
```
**Purpose**: Display referenced document sources
**Features**: 
- Clean markdown display
- Document names and page numbers
- No content excerpts, only attribution

### Event Handlers (Lines 297-320)

#### Document Processing (Lines 298-301)
```python
process_btn.click(
    process_documents,
    inputs=[file_input],
    outputs=[status_display, msg]
)
```
**Purpose**: Handle document processing button click
**Flow**: Button click → process_documents() → Update status and enable input

#### Message Submission (Lines 303-306)
```python
msg.submit(
    chat_response,
    inputs=[msg, chatbot],
    outputs=[msg, chatbot, sources_display]
)
```
**Purpose**: Handle Enter key in message input
**Flow**: Enter key → chat_response() → Clear input, update chat and sources

#### Send Button (Lines 308-311)
```python
submit_btn.click(
    chat_response,
    inputs=[msg, chatbot],
    outputs=[msg, chatbot, sources_display]
)
```
**Purpose**: Handle send button click
**Flow**: Button click → chat_response() → Clear input, update chat and sources

#### Clear Chat (Lines 313-315)
```python
clear_btn.click(
    clear_chat,
    outputs=[chatbot, msg]
)
```
**Purpose**: Handle clear chat button
**Flow**: Button click → clear_chat() → Empty chat and input

### App Launch (Lines 317-323)

```python
if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates public link
        debug=True,  # Enable debug mode
        show_error=True  # Show error messages
    )
```
**Purpose**: Launch the Gradio application
**Options**:
- `share=True`: Generate public URL for sharing
- `debug=True`: Enable debugging features
- `show_error=True`: Display error messages to users

---

## UI/UX Features

### Clean Interface Design
- **No Icons**: Minimal design without decorative icons
- **Professional Layout**: Three-column responsive design
- **Clear Typography**: Readable fonts and proper hierarchy
- **Subtle Styling**: Clean borders and spacing

### Sources Display
- **Document Names**: Shows actual uploaded filenames
- **Page Numbers**: Clear page attribution
- **Clean Format**: Numbered list without content excerpts
- **Professional Appearance**: Suitable for academic use

### Enhanced User Experience
- **Status Feedback**: Real-time processing status
- **Clear Instructions**: Step-by-step guidance
- **Responsive Layout**: Adapts to different screen sizes
- **Professional Styling**: Business-ready appearance

---

## Configuration Files

### requirements.txt
```
streamlit
python-dotenv
langchain==0.3.27
langchain-openai==0.3.35
langchain-community==0.3.17
PyPDF2
faiss-cpu
openai
sentence-transformers
gradio
```

**Purpose**: Python dependencies for the project
**Key Libraries**:
- `gradio`: Web interface framework
- `langchain-*`: AI/ML framework components
- `openai`: OpenAI API client
- `sentence-transformers`: Local embedding models
- `faiss-cpu`: Vector database
- `PyPDF2`: PDF processing

### .env (Environment Variables)
```
OPENAI_API_KEY=your_openai_api_key_here
```

**Purpose**: Store sensitive API keys securely
**Usage**: Loaded via `load_dotenv()` at startup

---

## Data Flow

### Enhanced Document Processing Flow:
1. User uploads PDFs via `file_input`
2. `process_documents()` extracts text and metadata using PyPDF2
3. Text is chunked with document information by `RecursiveCharacterTextSplitter`
4. Chunks with metadata are embedded using Hugging Face model
5. Embeddings stored in FAISS vector database with metadata
6. Status updated and chat enabled

### Enhanced Question-Answering Flow:
1. User submits question via `msg` textbox
2. `chat_response()` formats conversation history
3. FAISS performs similarity search on question
4. Relevant document chunks with metadata retrieved
5. Context + history + question sent to OpenAI
6. Response generated and displayed in chatbot
7. Sources extracted and formatted for display
8. Clean attribution shown in sources panel

---

## Error Handling

### Document Processing Errors:
- No files uploaded: "Please upload PDF files first."
- Empty PDFs: "No text found in the uploaded PDFs."
- Processing failures: "Error processing documents: [details]"

### Chat Response Errors:
- Unprocessed documents: "Please upload and process PDF documents first."
- API failures: "Sorry, I encountered an error: [details]"
- Vector store issues: Handled by try-catch blocks

---

## Performance Considerations

### Memory Management:
- Global variables prevent reloading vector store
- Conversation history limited to prevent context overflow
- Text chunks sized for optimal embedding performance
- Metadata efficiently stored with chunks

### Processing Speed:
- FAISS provides fast similarity search
- Hugging Face embeddings optimized for speed
- OpenAI API responses typically < 2 seconds
- Clean sources formatting is lightweight

### Storage:
- Vector store saved locally to avoid reprocessing
- FAISS index files compact and efficient
- No cloud storage required for documents
- Metadata embedded in vector store

---

## Security Features

### API Key Management:
- Environment variables prevent hardcoding
- .env file excluded from Git via .gitignore
- No API keys exposed in client-side code

### Data Privacy:
- Documents processed locally
- No data sent to third parties except API calls
- Vector store stored locally on disk
- No sensitive content in sources display

---

## Deployment Options

### Local Development:
```bash
conda activate dev_task
python gradio_app.py
```

### Hugging Face Spaces:
- Free hosting for Gradio apps
- GPU support available
- Direct GitHub integration

### Custom Hosting:
- Docker containerization
- Railway/Render deployment
- AWS/Azure enterprise options

---

## Recent Updates (Latest Version)

### UI/UX Improvements:
- **Clean Interface**: Removed all icons for professional appearance
- **Sources Panel**: Dedicated panel showing document names and page numbers
- **Three-Column Layout**: Document Manager, Chat, and Sources panels
- **Enhanced Metadata**: Better document tracking and attribution
- **Professional Styling**: Clean, business-ready design

### Technical Enhancements:
- **Metadata Preservation**: Document names preserved through processing
- **Clean Sources Display**: Formatted sources without content excerpts
- **Enhanced Error Handling**: Better user feedback and error messages
- **Improved Chat Flow**: Sources displayed alongside responses

### Code Quality:
- **Better Documentation**: Comprehensive inline comments
- **Modular Functions**: Separated concerns for better maintainability
- **Enhanced Processing**: Robust document processing with metadata
- **Clean Architecture**: Well-organized code structure

---

## Future Enhancement Points

### UI Improvements:
- Add progress bars for processing
- Implement file type expansion (Word, PowerPoint)
- Add theme customization options
- Implement responsive design improvements

### Feature Enhancements:
- Document summarization
- Flashcard generation
- Learning analytics dashboard
- Study session tracking
- Export chat history
- Bookmark important responses

### Technical Improvements:
- Streaming responses for better UX
- Advanced RAG strategies (compression, re-ranking)
- Multiple LLM model options
- Real-time collaboration features
- Batch document processing
- Advanced search filters

### Performance Optimizations:
- Caching for frequently asked questions
- Parallel processing for large documents
- Optimized chunking strategies
- Memory usage monitoring
