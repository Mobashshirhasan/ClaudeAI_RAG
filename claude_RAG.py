import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_anthropic import ChatAnthropic
import anthropic

#  Set your API keys
api_key = os.environ.get("ANTHROPIC_API_KEY")   
client = anthropic.Anthropic(api_key=api_key)

# Directory containing PDF files
pdf_directory = " "  # Enter your path,where your pdfs is stored

# Initialize an empty list to store all documents
all_documents = []

# Load all PDF files from the directory
for filename in os.listdir(pdf_directory):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_directory, filename)
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_documents.extend(documents)
            print(f"Successfully loaded: {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")

if not all_documents:
    raise ValueError("No PDF files were found or loaded successfully")

 # Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(all_documents)

# Use OpenAI Embeddings for the document processing
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# Create a FAISS index and store the embeddings
faiss_db = FAISS.from_documents(documents=chunks, embedding=embedding)

 # Save the FAISS index locally
index_name = "faiss_index_all_pdfs"
faiss_db.save_local(f"./{index_name}")

# Load the FAISS index from the disk
faiss_db = FAISS.load_local(
    folder_path=f"./{index_name}",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

# Set up the LLM for Anthropic Claude
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)

# Pull the prompt from the hub
prompt = hub.pull("rlm/rag-prompt")

#  Set up the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=faiss_db.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
)

def ask_question(question):
    """
    Function to ask questions about the PDF content
    """
    result = qa_chain({"query": question})
    return result["result"] if "result" in result else result

# Example usage
while True:
    user_question = input("\nAsk a question about your PDFs (or type 'exit' to quit): ")
    if user_question.lower() == 'exit':
        break
    
    try:
        response = ask_question(user_question)
        print("\nAnswer:", response)
    except Exception as e:
        print(f"Error processing question: {str(e)}")
