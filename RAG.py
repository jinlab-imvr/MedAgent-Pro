import warnings
import requests
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community")

MEDLINEPLUS_URLS = {
    "Atelectasis": [
        "https://medlineplus.gov/lungdiseases.html",
        "https://medlineplus.gov/breathingproblems.html",
    ],
    "Cardiomegaly": [
        "https://medlineplus.gov/heartdiseases.html",
        "https://medlineplus.gov/cardiomyopathy.html",
    ],
    "Consolidation": [
        "https://medlineplus.gov/pneumonia.html",
        "https://medlineplus.gov/lungdiseases.html",
    ],
    "Edema": [
        "https://medlineplus.gov/edema.html",
        "https://medlineplus.gov/heartfailure.html",
    ],
    "Enlarged Cardiomediastinum": [
        "https://medlineplus.gov/heartdiseases.html",
        "https://medlineplus.gov/aorticaneurysm.html",
    ],
    "Fracture": [
        "https://medlineplus.gov/fractures.html",
        "https://medlineplus.gov/chestinjuriesanddisorders.html",
    ],
    "Lung Lesion": [
        "https://medlineplus.gov/lungcancer.html",
        "https://medlineplus.gov/lungdiseases.html",
    ],
    "Lung Opacity": [
        "https://medlineplus.gov/lungdiseases.html",
        "https://medlineplus.gov/pneumonia.html",
    ],
    "Pleural Effusion": [
        "https://medlineplus.gov/pleuraldisorders.html",
        "https://medlineplus.gov/lungdiseases.html",
    ],
    "Pneumonia": [
        "https://medlineplus.gov/pneumonia.html",
        "https://medlineplus.gov/bacterialinfections.html",
    ],
    "Pneumothorax": [
        "https://medlineplus.gov/collapsedlung.html",
        "https://medlineplus.gov/pleuraldisorders.html",
    ],
    "Support Devices": [
        "https://medlineplus.gov/criticalcare.html",
        "https://medlineplus.gov/medicaldevicesafety.html",
    ],
}


class RAG_Module:
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-2024-11-20"):
        self.openai_api_key = openai_api_key
        self.model = model

        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a medical RAG expert. Based on the retrieved context, "
                "provide a concise clinical guideline for the queried condition.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Instructions:\n"
                "1. Summarise the key diagnostic indicators and clinical criteria.\n"
                "2. Focus on what can be assessed from chest X-ray images.\n"
                "3. List the indicators in a structured way.\n"
                "4. If the context is insufficient, state 'No relevant data found.'\n\n"
                "Answer:"
            ),
        )

    def _get_urls_for_finding(self, finding: str):
        """
        Select MedlinePlus URLs based on the finding name (metadata filtering).
        Falls back to generic lung disease page if finding is unknown.
        """
        urls = MEDLINEPLUS_URLS.get(finding)
        if urls:
            return urls
        # fuzzy match
        finding_lower = finding.lower()
        for key, val in MEDLINEPLUS_URLS.items():
            if key.lower() in finding_lower or finding_lower in key.lower():
                return val
        return ["https://medlineplus.gov/lungdiseases.html"]

    def _fetch_page_text(self, url: str):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                return soup.get_text(separator="\n", strip=True)
            else:
                print(f"[RAG] Failed to fetch {url} (status {resp.status_code})")
                return None
        except Exception as e:
            print(f"[RAG] Error fetching {url}: {e}")
            return None

    def _chunk_and_retrieve(self, texts, query, top_k=5):
        """
        Split texts into 300-token chunks, embed with OpenAI, retrieve top-k
        via FAISS similarity search.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
        )
        documents = []
        for text, url in texts:
            chunks = splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": url}))

        if not documents:
            return []

        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        vector_store = FAISS.from_documents(documents, embeddings)
        retrieved = vector_store.similarity_search(query, k=top_k)
        return retrieved

    def _build_qa_chain(self, retriever):
        llm = ChatOpenAI(
            model=self.model,
            openai_api_key=self.openai_api_key,
            temperature=1,
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.qa_prompt},
        )

    def query(self, finding: str, query: str = None):
        """
        Two-step RAG retrieval for a given chest X-ray finding.

        Args:
            finding: one of the 12 MIMIC findings (e.g. "Atelectasis")
            query:   optional custom query; defaults to
                     "How to diagnose {finding} from chest X-ray?"

        Returns:
            str: summarised guideline text from retrieved knowledge
        """
        if query is None:
            query = f"How to diagnose {finding} from chest X-ray images?"

        urls = self._get_urls_for_finding(finding)
        print(f"[RAG] Finding: {finding} | URLs: {urls}")

        texts = []
        for url in urls:
            page_text = self._fetch_page_text(url)
            if page_text:
                texts.append((page_text, url))

        if not texts:
            return "No relevant data found."

        retrieved_docs = self._chunk_and_retrieve(texts, query, top_k=5)
        print(f"[RAG] Retrieved {len(retrieved_docs)} chunks for '{finding}'")

        if not retrieved_docs:
            return "No relevant data found."

        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        all_texts = [(t, url) for t, url in texts]
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        all_docs = []
        for text, url in all_texts:
            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_docs.append(Document(page_content=chunk, metadata={"source": url}))

        vector_store = FAISS.from_documents(all_docs, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        qa_chain = self._build_qa_chain(retriever)
        result = qa_chain.invoke({"query": query})
        return result["result"]

    def query_all_findings(self, findings: list = None):
        """
        Run RAG for all (or specified) MIMIC findings.

        Returns:
            dict: {finding_name: rag_result_text}
        """
        if findings is None:
            findings = list(MEDLINEPLUS_URLS.keys())

        results = {}
        for finding in findings:
            print(f"\n{'='*60}")
            print(f"[RAG] Processing: {finding}")
            print(f"{'='*60}")
            results[finding] = self.query(finding)

        return results
