from ddgs import DDGS
from typing import List

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

print("ddgs_search results", ddgs_search("python application fields", max_results=3))

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

    #ddgs_runnable = RunnableLambda(lambda q: ddgs_search(q, max_results=5))
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
