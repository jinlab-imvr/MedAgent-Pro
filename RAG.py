import warnings
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Suppress LangChain and related warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community")


class RAG_Module:
    def __init__(self, openai_api_key, url_list=None):
        """
        Initialize the RAG (Retrieval-Augmented Generation) model with the OpenAI API key and a list of known URLs.
        
        Args:
            openai_api_key (str): OpenAI API key
            url_list (list, optional): the known URLs to retrieve information from
        """
        self.openai_api_key = openai_api_key
        self.url_list = url_list or [
            "https://medlineplus.gov/glaucoma.html",  
        ]
        # Define a custom prompt template for the QA model
        self.custom_qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a trusted medical and surgical Retrieval-Augmented Generation expert. Below is some context from your knowledge base, followed by a question.

            Context:
            {context}

            Question:
            {question}

            Instructions:
            1. Do NOT repeat or restate the ANY part question or answer in your answer.
            2. Just return relevant information and the context.
            3. If the context does not allow you to determine an answer, respond with "No relevant data found."

            Answer:
            """
        )

    def _fetch_raw_text(self, url):
        """
        Fetch the raw text content of a webpage.

        Args:
            url (str): URL of the webpage to fetch
            
        Returns:
            str or None: the raw text content of the webpage, or None if an error occurred
        """
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text()
                return text
            else:
                print(f"[ERROR] Failed to retrieve {url}")
                return None
        except Exception as e:
            print(f"[ERROR] Error fetching {url}: {e}")
            return None

    def _build_qa_chain(self, retriever):
        """
        Build a RetrievalQA chain for answering questions based on the given LangChain retriever.
        
        Args:
            retriever: LangChain retriever
            
        Returns:
            RetrievalQA 对象
        """
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key,
            temperature=0
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": self.custom_qa_prompt
            }
        )
        return qa_chain

    def query(self, query):
        """
        Query the RAG model with a user query and retrieve answers from the known URLs.

        Args:
            query (str): the user query to search for
            
        Returns:
            str: the formatted results of the query
        """
        results = {}

        for url in self.url_list:
            raw_text = self._fetch_raw_text(url)
            if raw_text is None:
                results[url] = "No relevant data found."
                continue

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            split_docs = text_splitter.split_text(raw_text)

            documents = [Document(page_content=chunk, metadata={"source": url}) for chunk in split_docs]

            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            vector_store = FAISS.from_documents(documents, embeddings)
            retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.1}
            )

            retrieved_docs = retriever.get_relevant_documents(query)
            print(f"Retrieved {len(retrieved_docs)} documents from {url}.")

            if not retrieved_docs:
                results[url] = "No relevant data found."
                continue

            qa_chain = self._build_qa_chain(retriever)
            result = qa_chain.invoke({"query": query})
            results[url] = result["result"]

        formatted_results = "\n\n".join([f"{url}:\n{answer}" for url, answer in results.items()])
        return formatted_results



