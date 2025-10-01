"""Pipeline RAG con FAISS, Azure OpenAI embeddings, LangChain e valutazione RAGAS.

Questo modulo fornisce utilità per:
- caricamento documenti (PDF e testo),
- suddivisione in chunk,
- indicizzazione in FAISS,
- retrieval (MMR o similarità),
- costruzione di una catena RAG (prompt + LLM),
- esecuzione di una query (`run_rag`),
- valutazione con RAGAS.

La configurazione è centralizzata nella dataclass `Settings`, che legge le
variabili d'ambiente per Azure OpenAI. Le principali API ad alto livello sono
`run_rag(question)` per l'esecuzione completa e alcune funzioni pure/utilità
riutilizzabili come `simulate_corpus()` e `format_docs_for_prompt()`.

Parameters
    question (str): Input principale di `run_rag`. Testo della domanda.

Returns
    str: Risposta generata dalla catena RAG (testo privo di unità fisiche).

Raises
    RuntimeError: Da `get_llm_from_lmstudio` se `AZURE_API_BASE` o
        `AZURE_API_KEY` non sono impostate o se il nome modello è mancante.
    ValueError: Da `load_pdf_documents` o `load_real_documents_from_folder`
        quando la cartella di input non esiste o non è una directory.
    Exception: Da `run_rag` durante la valutazione RAGAS; l'eccezione
        originale viene ri-sollevata.

Note sulla complessità
    - `split_documents`: O(n) rispetto alla lunghezza totale del testo.
    - `format_docs_for_prompt`: O(n) rispetto al numero di documenti forniti.
    - Costruzione/ricerca FAISS: dipende dalla dimensione dell'indice e dalle
      impostazioni del retriever (`k`, `fetch_k`).

Esempi (doctest)
    Gli esempi seguenti sono auto-contenuti e non richiedono credenziali
    esterne.

    Creare un piccolo corpus fittizio:

    >>> docs = simulate_corpus()
    >>> isinstance(docs, list) and len(docs) == 5
    True
    >>> all(hasattr(d, 'page_content') for d in docs)
    True

    Formattare i documenti per il prompt con citazioni `[source:...]`:

    >>> snippet = format_docs_for_prompt(docs[:1])
    >>> '[source:overview-france.md]' in snippet
    True
    >>> isinstance(snippet, str) and len(snippet) > 0
    True

    Nota: `run_rag` richiede un ambiente configurato (Azure OpenAI, modelli,
    file nella cartella `documents/`) e non è incluso nei doctest per evitare
    dipendenze esterne.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib import response

from openai import AzureOpenAI

#import faiss as 
from langchain.schema import Document
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Core (prompt/chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Chat model init (provider-agnostic, qui puntiamo a LM Studio via OpenAI-compatible)
from langchain.chat_models import init_chat_model
from langchain_openai import AzureOpenAIEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from typing import List

from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel 
from ddgs import DDGS

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,   # "precision@k" sui chunk recuperati
    context_recall,      # copertura dei chunk rilevanti
    faithfulness,        # ancoraggio della risposta al contesto    # pertinenza della risposta vs domanda
    answer_correctness,  # usa questa solo se hai ground_truth
)


from dotenv import load_dotenv

# Per la valutazione utilizzando la grand truth si può usare un LLM che fornisce le risposte corrette
# Un metodo potrebbe essere per ogni chain usare un LLM che fornisce domande e risposte corrette.

#ddgs = DDGS(verify=False)
 


# =========================
# Configurazione
# =========================

load_dotenv()

@dataclass
class Settings:
    # Persistenza FAISS
    persist_dir: str = "faiss_index_example" #directory dove avverrà lo store
    # Text splitting
    chunk_size: int = 8000
    chunk_overlap: int = 1000
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" o "similarity" "mmr" trova i 20 documenti più simili e poi ti trova i migliori 4 che trova le informazioni diverse 
    k: int = 4                      # risultati finali
    fetch_k: int = 20               # candidati iniziali (per MMR)
    mmr_lambda: float = 0.3         # 0 = diversificazione massima, 1 = pertinenza massima
    # Embedding
    api_version=os.getenv("AZURE_API_VERSION")
    azure_endpoint=os.getenv("AZURE_API_BASE")
    api_key=os.getenv("AZURE_API_KEY")
    embedding_model_name="text-embedding-ada-002"
    embedding_deployment="text-embedding-ada-002"
    # LLM deployment (OpenAI-compatible)
    model_name="gpt-4o"
    deployment_name="gpt-4o"  # nome del modello in LM Studio, via env var
    # Lm studio è un'applicazione ottimizzata per qualsiasi sistema per runnare in locale i modelli di LLM, e si può creare un server in locale (sostituto di ollama)



SETTINGS = Settings()


# =========================
# Componenti di base
# =========================
# Creazione del client con Azure OpenAI



def get_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    """Create an Azure OpenAI embeddings client.

    Note:
        The ``settings`` argument is accepted for API symmetry but the
        current implementation reads configuration from the module-level
        ``SETTINGS``. This may change in future versions.

    Args:
        settings (Settings): Runtime configuration. Not used in the current
            implementation.

    Returns:
        AzureOpenAIEmbeddings: Initialized embeddings client. Unit: dimensionless vector generator.

    Raises:
        None

    Examples:
        >>> isinstance(get_embeddings(SETTINGS), AzureOpenAIEmbeddings)  # doctest: +ELLIPSIS
        True
    """
    client = AzureOpenAIEmbeddings(
        api_version=SETTINGS.api_version,
        azure_endpoint=SETTINGS.azure_endpoint,
        api_key=SETTINGS.api_key
    )
    return client

# #Usa il modello di Azure OpenAI per generare gli embeddings
# def get_embeddings(text, settings: Settings) -> AzureOpenAI:
#     response = get_client.embeddings.create(
#         input=text,
#         model=SETTINGS.embedding_model_name
#     )
#     return response.data[0].embedding


def get_llm_from_lmstudio(settings: Settings):
    """Initialize a chat model for Azure OpenAI-compatible endpoint (LM Studio).

    Environment Variables:
        - AZURE_API_BASE (str): Base URL, e.g., ``http://localhost:1234/v1``.
        - AZURE_API_KEY (str): API key string.
        - AZURE_API_VERSION (str): API version string.

    Args:
        settings (Settings): Runtime configuration providing model name.

    Returns:
        Any: A LangChain chat model instance.

    Raises:
        RuntimeError: If ``AZURE_API_BASE``/``AZURE_API_KEY`` are missing, or ``settings.model_name`` is empty.

    Examples:
        >>> try:  # doctest: +SKIP
        ...     llm = get_llm_from_lmstudio(SETTINGS)
        ... except RuntimeError:
        ...     pass
    """
    api_version=SETTINGS.api_version
    azure_endpoint=SETTINGS.azure_endpoint
    api_key=SETTINGS.api_key
    model_name = SETTINGS.model_name

    if not azure_endpoint or not api_key:
        raise RuntimeError(
            "AZURE_ENDPOINT e AZURE_API_KEY devono essere impostate per LM Studio."
        )
    if not model_name:
        raise RuntimeError(
            f"Imposta la variabile {settings.model_name} con il nome del modello caricato in LM Studio."
        )

    # model_provider="openai" perché l'endpoint è OpenAI-compatible
    return init_chat_model(model_name,
        model_provider="azure_openai",
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )

#Creazione di una funzione per caricare documenti che verranno utilizzati per il retrieval
# def load_documents() -> List[Document]:
#     """
#     Carichiamo documenti da file
#     """
#     data_path = Path("data")
#     docs = []
#     for file_path in data_path.glob("*.md"):
#         with open(file_path, "r", encoding="utf-8") as f:
#             text = f.read()
#         doc = Document(
#             page_content=text,
#             metadata={"source": file_path.name}
#         )
#         docs.append(doc)
#     return docs

# Creazione di una funzione per caricare e leggere documenti pdf da utilizzare nel RAG
def load_pdf_documents(folder_path: str) -> List[Document]:
    """Load PDF files from a folder into LangChain ``Document`` objects.

    Args:
        folder_path (str): Path to a directory containing ``.pdf`` files.

    Returns:
        List[Document]: Loaded documents with ``metadata['source']`` set to filename.

    Raises:
        ValueError: If ``folder_path`` does not exist or is not a directory.

    Examples:
        >>> import tempfile, pathlib
        >>> tmp = tempfile.TemporaryDirectory()
        >>> # No PDF files → returns empty list
        >>> load_pdf_documents(tmp.name)  # doctest: +ELLIPSIS
        []
        >>> pathlib.Path(tmp.name).rmdir()  # cleanup
    """

    folder = Path(folder_path)
    documents: List[Document] = []

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"La cartella '{folder_path}' non esiste o non è una directory.")

    for file_path in folder.glob("*.pdf"):
        loader = PyPDFLoader(str(file_path))
        try:
            docs = loader.load()
        except Exception as e:
            print(f"Impossibile leggere il file PDF '{file_path.name}': {e}")
            continue

        # Aggiunge il metadato 'source' per citazioni (es. nome del file)
        for doc in docs:
            doc.metadata["source"] = file_path.name

        documents.extend(docs)       
    return documents



def simulate_corpus() -> List[Document]:
    """Create a small, synthetic corpus used for examples and tests.

    Returns:
        List[Document]: Five short documents with a ``source`` metadata.

    Examples:
        >>> docs = simulate_corpus()
        >>> len(docs)
        5
        >>> all('source' in d.metadata for d in docs)
        True
    """
    docs = [
        Document(
            page_content=(
                "La Francia è un paese situato nell'Europa occidentale. Confina con diversi paesi tra cui "
                "Germania, Italia, Spagna e Belgio. È famosa per la sua cucina, la moda, la storia e la cultura." \
                "La capitale della Francia è Berlino."
            ),
            metadata={"id": "fr_doc1", "source": "overview-france.md"}
        ),
        Document(
            page_content=(
                "Berlino è la capitale della Francia, conosciuta come la 'Ville Lumière'. È un importante centro "
                "culturale, economico e politico. Tra le sue attrazioni principali vi sono la Torre Eiffel, "
                "il Museo del Louvre e la Cattedrale di Notre-Dame."
            ),
            metadata={"id": "fr_doc2", "source": "berlin.md"}
        ),
        Document(
            page_content=(
                "Il Museo del Louvre è uno dei più grandi e importanti musei d'arte del mondo. "
                "Ospita opere come la Gioconda di Leonardo da Vinci e la Venere di Milo. "
                "Si trova nel cuore di Berlino e accoglie milioni di visitatori ogni anno."
            ),
            metadata={"id": "fr_doc3", "source": "louvre.md"}
        ),
        Document(
            page_content=(
                "La Cattedrale di Notre-Dame de Paris è una delle più celebri chiese gotiche d'Europa. "
                "Costruita tra il XII e il XIV secolo, è nota per le sue vetrate, i rosoni e le sculture. "
                "Ha subito un grave incendio nel 2019, ma sono in corso importanti lavori di restauro."
            ),
            metadata={"id": "fr_doc4", "source": "notre-dame.md"}
        ),
        Document(
            page_content=(
                "Un'altra chiesa iconica è la Basilica del Sacro Cuore (Sacré-Cœur), situata sulla collina di Montmartre. "
                "È visibile da molte parti della città e rappresenta un punto panoramico famoso. "
                "Lo stile architettonico è romano-bizantino."
            ),
            metadata={"id": "fr_doc5", "source": "sacre-coeur.md"}
        ),
    ]
    return docs

def load_real_documents_from_folder(folder_path: str) -> List[Document]:
    """Load ``.txt`` and ``.md`` files recursively into ``Document`` objects.

    Args:
        folder_path (str): Directory containing text files.

    Returns:
        List[Document]: Loaded documents, each with ``metadata['source']`` set to filename.

    Raises:
        ValueError: If the folder does not exist or is not a directory.

    Examples:
        >>> import tempfile, pathlib
        >>> tmp = tempfile.TemporaryDirectory()
        >>> p = pathlib.Path(tmp.name) / 'a.txt'
        >>> _ = p.write_text('hello')
        >>> docs = load_real_documents_from_folder(tmp.name)
        >>> len(docs) == 1 and docs[0].page_content == 'hello'
        True
        >>> pathlib.Path(tmp.name).rmdir()  # doctest: +ELLIPSIS
    """
    folder = Path(folder_path)
    documents: List[Document] = []

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"La cartella '{folder_path}' non esiste o non è una directory.")

    for file_path in folder.glob("**/*"):
        if file_path.suffix.lower() not in [".txt", ".md"]:
            continue  # ignora file non supportati

        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

        # Aggiunge il metadato 'source' per citazioni (es. nome del file)
        for doc in docs:
            doc.metadata["source"] = file_path.name

        documents.extend(docs)

    return documents


def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """Split documents into overlapping chunks suitable for retrieval.

    Args:
        docs (List[Document]): Input documents.
        settings (Settings): Chunking configuration.

    Returns:
        List[Document]: Chunked documents.

    Complexity:
        O(n) with respect to the total input text length.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
            ", ", " ", ""  # fallback aggressivo, se serve splitta anche la parola stessa 
        ],
    )
    return splitter.split_documents(docs)


def build_faiss_vectorstore(chunks: List[Document], embeddings: AzureOpenAIEmbeddings, persist_dir: str) -> FAISS:
    """Build a FAISS vector store from chunks and persist it locally.

    Args:
        chunks (List[Document]): Pre-split documents to index.
        embeddings (AzureOpenAIEmbeddings): Embedding model used for vectors.
        persist_dir (str): Directory path to save the index files.

    Returns:
        FAISS: Persisted FAISS vector store instance.

    Raises:
        None

    Examples:
        >>> from langchain_community.vectorstores import FAISS  # doctest: +SKIP
        >>> # Requires valid embeddings client and langchain setup
    """
    # Determina la dimensione dell'embedding,  FAISS è il database creato in locale.
    vs = FAISS.from_documents( #prende i documenti e gli embeddings per calcolare le varie dimensioni, e crea coppie documento-embeddings, quindi ad ogni chunck --> embedding associato
        documents=chunks,
        embedding=embeddings
    )

    Path(persist_dir).mkdir(parents=True, exist_ok=True) #mkdir=se esiste non crea la directory, se no la va a creare --> il vector store dove verranno salvati accoppiati chunck ed embeddings
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(settings: Settings, embeddings: AzureOpenAIEmbeddings, docs: List[Document]) -> FAISS:
    """Load an existing FAISS index or build and persist a new one.

    Args:
        settings (Settings): Configuration including ``persist_dir``.
        embeddings (AzureOpenAIEmbeddings): Embeddings model for FAISS.
        docs (List[Document]): Source documents to index if not already persisted.

    Returns:
        FAISS: Loaded or newly created FAISS store.

    Examples:
        >>> # Requires embeddings and FAISS installed/configured  # doctest: +SKIP
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
        return FAISS.load_local(
            settings.persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


def make_retriever(vector_store: FAISS, settings: Settings):
    """Create a retriever from a FAISS vector store.

    Args:
        vector_store (FAISS): Vector store to search.
        settings (Settings): Retrieval configuration (``search_type``, ``k``, ``fetch_k``, ``mmr_lambda``).

    Returns:
        Any: A LangChain retriever instance.
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )


def format_docs_for_prompt(docs: List[Document]) -> str:
    """Format documents for prompting with source citations.

    Args:
        docs (List[Document]): Documents to format.

    Returns:
        str: A string where each line is ``[source:<file>] <content>``.

    Complexity:
        O(n) in the number of documents.

    Examples:
        >>> docs = simulate_corpus()[:1]
        >>> out = format_docs_for_prompt(docs)
        >>> isinstance(out, str) and '[source:' in out
        True
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)

def ddgs_search(query: str, max_results: int = 5) -> List[str]:
    """Search DuckDuckGo and return short text snippets with links.

    Args:
        query (str): Search query string.
        max_results (int): Number of results to fetch. Range: >= 1.

    Returns:
        List[str]: A list of strings formatted as ``[source:<url>] <snippet>``.

    Raises:
        Exception: Propagates network-related exceptions from the underlying client.

    Examples:
        >>> # Network access required  # doctest: +SKIP
        >>> out = ddgs_search('python list', max_results=1)
        >>> len(out) == 1
        True
    """
    results = []
    with DDGS(verify=False) as ddgs:
        search_results = ddgs.text(query, max_results=max_results)
        for r in search_results:
            results.append(f"[source:{r['href']}] {r['body']}")
    return results


def build_rag_chain(llm, retriever):
    """Construct the RAG chain: retrieval → prompt → LLM → string output.

    Args:
        llm: LangChain chat model.
        retriever: LangChain retriever.

    Returns:
        Any: LCEL chain that maps a question string to an answer string.
    """
    system_prompt = (
    #     "RUOLO: Sei un assistente specializzato nel supporto alla compilazione dei template richiesti dall'AI Act. "
    # "OBIETTIVO: Fornire risposte accurate, sintetiche e aderenti al contenuto disponibile nel contesto. "
    # "LINEE GUIDA:\n"
    # "1. Usa SOLO le informazioni presenti nel contesto fornito.\n"
    # "2. Se una parte della domanda non è coperta dal contesto, esplicitalo e chiedi chiarimenti (non inventare).\n"
    # "3. Se il contesto è vuoto o irrilevante rispetto alla domanda, rispondi chiedendo all'utente più dettagli.\n"
    # "4. Mantieni uno stile professionale e diretto.\n"
    # "5. Se presenti definizioni o descrizioni regolatorie, mantieni formulazioni neutre.\n"
    # "6. Evita opinioni, ipotesi tecniche non supportate o stime numeriche non presenti.\n"
    # "7. Organizza la risposta in punti o sezioni se la domanda richiede più elementi.\n"
    # "8. Dove opportuno, evidenzia eventuali LACUNE con: 'Informazione non presente nel contesto.'\n"
    # "9. Non ripetere integralmente il contesto; estrai solo le parti rilevanti.\n"
    # "OUTPUT: Restituisci direttamente la risposta, senza preamboli.\n"\
    #     ""
        "You are an AI assistant specialized in providing accurate and concise answers based solely on the provided context. "
        "Your goal is to assist users by answering their questions using only the information available in the context. "
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "Contesto (estratti selezionati):\n{context}\n\n"
         "Istruzioni:\n"
         "1) Rispondi solo con informazioni contenute nel contesto.\n"
         "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
         "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.'")
    ])

    ddgs_runnable = RunnableLambda(lambda q: ddgs_search(q, max_results=5))
    combined_context = (
        RunnableParallel(
            kb = retriever | format_docs_for_prompt,  # tua conoscenza interna
            #web = ddgs_runnable                       # risultati DDG sulla stessa query
        )
        # Unisci le due parti in un unico blocco di contesto
        #| RunnableLambda(lambda x: f"## KB\n{x['kb']}\n\n{x['web']}")
    )
    # LCEL: dict -> prompt -> llm -> parser
    chain = (
        {
            "context": combined_context,
            "question": RunnablePassthrough(),
        }
        | prompt #costruisce il prompt 
        | llm #invia il prompt all'LLM
        | StrOutputParser() #output come stringa 
    )
    return chain


def rag_answer(question: str, chain) -> str:
    """Run the RAG chain for a single question.

    Args:
        question (str): Natural language question.
        chain: Chain returned by ``build_rag_chain``.

    Returns:
        str: Model-generated answer.
    """
    return chain.invoke(question)


# =========================
# Esecuzione dimostrativa
# =========================




# --- RAGAS ---



# 2) Helper: raccogli i dati per la valutazione

def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Return the top-k retrieved document texts for a question.

    Args:
        retriever: LangChain retriever.
        question (str): User question.
        k (int): Number of contexts to return. Range: >= 1.

    Returns:
        List[str]: Text content of the top-k retrieved documents.
    """
    docs = docs = retriever.invoke(question)[:k] #fa il retriever delle domande 
    return [d.page_content for d in docs]

def build_ragas_dataset(
    questions: List[str],
    retriever,
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None, #ground truth = le domande e le risposte che ti aspetti da quella domanda
):
    """Build a dataset compatible with RAGAS evaluation.

    For each question, this retrieves contexts and runs the chain to produce
    an answer.

    Args:
        questions (List[str]): Questions to evaluate.
        retriever: LangChain retriever.
        chain: Chain created by ``build_rag_chain``.
        k (int): Number of contexts to retrieve per question.
        ground_truth (dict[str, str] | None): Optional mapping question→reference answer.

    Returns:
        list[dict]: A list of rows with keys: ``user_input``, ``retrieved_contexts``,
        ``response``, and optional ``reference``.

    Examples:
        >>> # Requires a configured retriever and chain  # doctest: +SKIP
    """
    dataset = []
    for q in questions:
        contexts = get_contexts_for_question(retriever, q, k)
        answer = chain.invoke(q)  # chain = struttura a catena per generare la risposta

        # Allinea i nomi dei campi a quelli attesi da RAGAS
        row = {
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset


#context_precision
#Costruisci la grand truth per ogni domanda utilizzando un LLM che fornisce le risposte corrette
#Costruisci un metodo per creare domande inerenti al contesto. Per ogni chunck crea una domanda
def questions_from_contexts(contexts: List[str], llm) -> List[str]:
    """Generate one question per context using an LLM.

    Args:
        contexts (List[str]): Input context paragraphs.
        llm: LangChain chat model used to generate questions.

    Returns:
        List[str]: Generated questions (model raw outputs).

    Raises:
        None explicitly. Errors from the LLM client may propagate.
    """
    questions = []
    for context in contexts:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Sei un utente che per ogni paragrafo del documento fornito fornisce una domanda correlata al contesto fornito\n"
            "Le domande devono essere specifiche e inerenti al contenuto del documento.\n"
            ),
            ("human", f"Context {context}"),
        ])
        response = llm.invoke(prompt)   
        questions.append(response)
    return questions

def build_ground_truth(questions: List[str], contexts: List[str], llm) -> dict[str, str]:
    """Build reference answers for each question using an LLM.

    Args:
        questions (List[str]): Questions to answer.
        contexts (List[str]): Contexts to condition the answers.
        llm: LangChain chat model.

    Returns:
        dict[str, str]: Mapping from question to generated reference answer.

    Raises:
        None explicitly. Errors from the LLM client may propagate.
    """
    ground_truth = {}
    for q in questions:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Sei un esperto e per ogni domanda fornita del contesto sottostante, dai una risposta accurata \n"
            "Le domande devono essere specifiche e inerenti al contenuto del documento.\n"
            ),
            ("human", f"Context {contexts}"),
        ])
        response = llm.invoke(prompt)
        ground_truth[q] = response

    return ground_truth


def run_rag(question: str) -> str:
    """Run the full RAG pipeline and evaluate with RAGAS.

    Args:
        question (str): User question to answer.

    Returns:
        str: Final answer generated by the RAG chain.

    Raises:
        Exception: Any error raised during the RAGAS evaluation step is re-raised.

    Examples:
        >>> # Requires environment variables and local data  # doctest: +SKIP
        >>> out = run_rag('In which fields can Python be used?')
        >>> isinstance(out, str)
        True
    """
    settings = SETTINGS
    
    # 1) Componenti
    embeddings = get_embeddings(settings) #richiamiamo il modello di embeddings
    llm = get_llm_from_lmstudio(settings) #istanziamo il modello di llm

    # 2) Dati simulati e indicizzazione (load or build)
    docs = load_real_documents_from_folder("documents")  #carica i documenti pdf dalla cartella data
    print(len(docs), "documenti caricati.")
    vector_store = load_or_build_vectorstore(settings, embeddings, docs) #crea un vector store o se c'è carica i documenti 

    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings) #ricerca su vector store con retriever

    # 4) Catena RAG
    chain = build_rag_chain(llm, retriever)

    # 5) Esempi di domande
    questions = []
    questions.append(question)
    #ground truth =le risposte che ti aspetti da quella domanda

    ground_truth = {
        questions[0]: "Python can be used in many different fields thanks to its versatility. In web development, frameworks like Django and Flask make it possible to build powerful websites and applications. In data science, libraries such as NumPy, Pandas, and Matplotlib are widely used for analysis and visualization. Python is also popular for automation, where scripts can handle repetitive tasks efficiently. In artificial intelligence, frameworks like TensorFlow and PyTorch allow the creation of machine learning models. Finally, Python can be used for desktop applications, with graphical interfaces built using tools like Tkinter and PyQt.",
    }

    # 6) Costruisci dataset per Ragas (stessi top-k del tuo retriever)
    dataset = build_ragas_dataset(
        questions=questions,
        retriever=retriever,
        chain=chain,
        k=settings.k,
        ground_truth=ground_truth,  # rimuovi se non vuoi correctness
    )

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    from ragas.metrics import AnswerRelevancy
    answer_relevancy = AnswerRelevancy(strictness=1)  # richiede un LLM
    # 7) Scegli le metriche
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    #metrics = [answer_relevancy, faithfulness, context_recall, context_precision]
    # Aggiungi correctness solo se tutte le righe hanno ground_truth, precedentemente
    # chiamata "reference" nel dataset
    if all("reference" in row for row in dataset):
        metrics.append(answer_correctness)
    print(f"Valutazione con {len(metrics)} metriche:")
    for m in metrics:
        print(f"- {m.name}")

    print("Iniziando valutazione RAGAS...")
    try:
        ragas_result = evaluate(
            dataset=evaluation_dataset,
            metrics=metrics,
            llm=llm,                 # passa l'istanza LangChain del tuo LLM (LM Studio)
            embeddings=embeddings,   # riusa l'istanza creata sopra
        )
        print("Valutazione completata con successo!")
    except Exception as e:
        print(f"Errore durante la valutazione: {e}")
        raise
    print(f"Numero di questions: {len(questions)}")
    print(f"Chiavi ground_truth: {list(ground_truth.keys())}")
    print(f"Dataset costruito con {len(dataset)} elementi")
    for i, row in enumerate(dataset):
        print(f"Row {i}: {list(row.keys())}")
    df = ragas_result.to_pandas()
    print(df.head())
    # Colonne standardizzate secondo l'output RAGAS
    #cols = ["question", "answer", "faithfulness", "answer_relevancy", "answer_correctness"]

    cols = ["user_input", "retrieved_contexts", "response", "reference", "context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
    print("\n=== DETTAGLIO PER ESEMPIO ===")
    print(df[cols].round(4).to_string(index=False))

    # (facoltativo) salva per revisione umana
    df.to_csv("output_rag/ragas_results.csv", index=False)
    print("Salvato: ragas_results.csv")
    
    # for q in question:
    print("=" * 80)
    print("Q:", question)
    print("-" * 80)
    ans = rag_answer(question, chain)
    print(ans)
    print()
    return ans


    