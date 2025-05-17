import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():
    """Loads the vector store with embedding model."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    return db


def set_custom_prompt(custom_prompt_template):
    """Sets the custom prompt template for the LLM."""
    return PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )


def load_llm(huggingface_repo_id, HF_TOKEN):
    """Loads the Hugging Face model."""
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"},
    )


def main():
    # Page configuration and branding
    st.set_page_config(page_title="Chikitsha Bot", page_icon="🩺", layout="centered")

    # Sidebar with information and navigation
    with st.sidebar:
        st.title("🩺 Chikitsha Bot")
        st.markdown("**Your AI Health Assistant**")
        st.markdown(
            "This bot can answer medical questions based on its training context."
        )
        st.markdown("Model: `Mistral-7B-Instruct`")
        st.markdown("VectorDB: FAISS + MiniLM-L6-v2")
        st.markdown("---")
        st.markdown("⚠️ This is not a replacement for professional medical advice.")

    # Main title
    st.title("ASK ME CHIKITSHA_BOT IS HERE ")

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # User input for prompt
    prompt = st.chat_input("Ask a medical question:")

    if prompt:
        # Show user's message
        st.chat_message("user").markdown(f"👤 **You:**\n{prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Define the custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer the user's question.
                If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
                Don't provide anything outside of the given context.

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            # Show a spinner while waiting for a response
            with st.spinner("🤖 Chikitsha is thinking..."):
                # Load the vector store and check for issues
                vectorstore = get_vectorstore()
                if not vectorstore:
                    st.error("Failed to load the vector store")
                    return

                # Create the RetrievalQA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(
                        huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN
                    ),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
                    },
                )

                # Get the response from the model
                response = qa_chain.invoke({"query": prompt})

                result = response["result"]
                source_documents = response["source_documents"]
                result_to_show = (
                    result
                    + "\n\n**Source Documents:**\n"
                    + "\n".join(
                        [
                            f"Doc {i+1}: {doc.page_content}"
                            for i, doc in enumerate(source_documents)
                        ]
                    )
                )

                # Show assistant's response
                st.chat_message("assistant").markdown(
                    f"🤖 **Chikitsha Bot:**\n{result_to_show}"
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": result_to_show}
                )

        except Exception as e:
            # Improved error handling
            st.error(f"An error occurred: {str(e)}")
            st.warning("Please try again later or contact support.")

    # Footer with optional help and feedback
    st.markdown(
        "<hr style='border: 2px solid #4CAF50'><center>Made with ❤️ by Ashutosh Chaudhary. Not a substitute for medical advice. "
        "Contact Support: <a href='mailto:ashutosh.chaudhary790@gmail.com'>ashutosh.chaudhary790@gmail.com</a></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
