from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

""" steps:
    1.load raw pdf(s) :from book The gale of encycopedia
    2.create chunks:splitting small parts of the book data 
    3.create vector embedding : 
    4.store embedding in FAISS"""


# 1
DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents= loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
# print("Length of the PDF pages: ",len(documents))


# 2

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50  
    )
    text_chunks= text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
# print("Length of Text Chunks: ",len(text_chunks))


# 3.
# sentence-transformers/all-MiniLM-L6-v2: it maps sentence and paragraph to 384 dimension dense vector space and can be used for clustering and semantic search.

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model
embedding_model=get_embedding_model()

# 4.

DB_FAISS_PATH="vectorstore/db_faiss"  #df ke place mao db likhn atha mistake ho gya
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)
