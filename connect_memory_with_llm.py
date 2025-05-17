import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. setup llm(mistral with HUGGINGFACE)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def load_llm(HUGGINGFACE_REPO_ID):
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,  # temperature 0.1 rakhenge to ans bhut prescide or corect ayega ek dum point to poin,word count increse krne ke liea hmlog 0.5 kr rhe hai
        model_kwargs={"token": HF_TOKEN, #kwargs=keyword argumnet
                      "max_length": "512"},
    )
    return llm


# 2.connect llm with FAISS and create chain
# set a custom prompt:means ismai ek custom prompt dete hai or context dete hai uske basis pe llm respond karta hai.


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}    # the context retrive from the database or llm to 
Question: {question}  # here all th query will go

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_tempelate):
    prompt= PromptTemplate(template=custom_prompt_tempelate,
                           input_variable=["context","question"])
    return prompt

# load databse :jo emvedding kis model ne embbed kre hai
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)
# allow_dangerous_deserialization (INTRODUCED BY LANGCHAIN FOR SAFETY AND SECURITY): if we have trust on sourse of information


# CREATE QA CHAIN
qa_chain=RetrievalQA.from_chain_type(
    llm= load_llm( HUGGINGFACE_REPO_ID),
    chain_type= "stuff",
    retriever =db.as_retriever(search_kwargs={'k':3}),
    return_source_documents= True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)


# Active /invoke chain with a single query


user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query':user_query})
print("RESUlt: ",response["result"])
print("SOURCE DOCUMENTS: ",response["source_documents"])
