import rag_qdrant_hybrid

import sys
from pathlib import Path

# Add the parent directory to the path to import from tools
sys.path.append(str(Path(__file__).parent.parent))

import os
from typing import List

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,   # "precision@k" sui chunk recuperati
    context_recall,      # copertura dei chunk rilevanti
    faithfulness,        # ancoraggio della risposta al contesto
    answer_relevancy,    # pertinenza della risposta vs domanda
    answer_correctness,  # usa questa solo se hai ground_truth
)

CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIRECTORY_PATH = os.path.dirname(CURRENT_FILE_PATH)

def build_ragas_dataset(
    questions: List[str],
    chain,
    client,
    s,
    embeddings,
    llm,
    ground_truth: dict[str, str] | None = None,
):
    """
    Esegue la pipeline RAG per ogni domanda e costruisce il dataset per Ragas.
    Ogni riga contiene: question, contexts, answer, (opzionale) ground_truth.
    """
    dataset = []
    for q in questions:
        # answer = chain.invoke(q)
        contexts = rag_qdrant_hybrid.hybrid_search(client, s, q, embeddings)

        # For Ragas: list of individual context strings
        context_texts = rag_qdrant_hybrid.extract_context_texts(contexts)

        # For the LLM chain: formatted concatenated string
        ctx = rag_qdrant_hybrid.format_docs_for_prompt(contexts)
        chain = rag_qdrant_hybrid.build_rag_chain(llm)
        answer = chain.invoke({"question": q, "context": ctx})

        print("ctx: ", ctx)

        row = {
            # chiavi richieste da molte metriche Ragas
            "user_input": q,
            "retrieved_contexts": context_texts,  # List of strings for Ragas
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset


def main():
    s = rag_qdrant_hybrid.SETTINGS
    
    s.collection = "documents"
    
    embeddings = rag_qdrant_hybrid.get_embeddings(s)
    llm = rag_qdrant_hybrid.get_llm(s)  # opzionale

    # 1) Client Qdrant
    client = rag_qdrant_hybrid.get_qdrant_client(s)

    # 2) Dati -> chunk
    # docs = simulate_corpus()
    docs_dir = os.path.join(CURRENT_DIRECTORY_PATH, "..\documents")
    docs = rag_qdrant_hybrid.load_real_documents_from_folder(docs_dir)  # cartella con PDF e MD
    chunks = rag_qdrant_hybrid.split_documents(docs, s)

    # 3) Crea (o ricrea) collection
    vector_size = len(embeddings.embed_query("test"))

    rag_qdrant_hybrid.recreate_collection_for_rag(client, s, vector_size)

    # 4) Upsert chunks
    rag_qdrant_hybrid.upsert_chunks(client, s, chunks, embeddings)
    
    questions = ["Qual è l'obiettivo O1 dell'intervento turismo cloud avs in Sardegna?", ""]
    
    chain = rag_qdrant_hybrid.build_rag_chain(llm)
    
    ground_truth = {
        questions[0]: "L'obiettivo O1 dell'intervento turismo cloud avs in Sardegna è sostenere fino al subentro di fornitori l'operatività dei sistemi informativi", 
        questions[1]: "Non ci sono informazioni rilevanti nei documenti caricati.",

    }
    
    # 6) Costruisci dataset per Ragas (stessi top-k del tuo retriever)
    dataset = build_ragas_dataset(
        questions=questions,
        chain=chain,
        client=client,
        s=s,
        embeddings=embeddings,
        llm=llm,
        ground_truth=ground_truth,  # rimuovi se non vuoi correctness
    )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # 7) Scegli le metriche
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    # Aggiungi correctness solo se tutte le righe hanno ground_truth
    if all("ground_truth" in row for row in dataset):
        metrics.append(answer_correctness)

    # 8) Esegui la valutazione con il TUO LLM e le TUE embeddings
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,                 # passa l'istanza LangChain del tuo LLM (LM Studio)
        embeddings=rag_qdrant_hybrid.get_embeddings(s),  # o riusa 'embeddings' creato sopra
    )

    df = ragas_result.to_pandas()
    cols = ["user_input", "response", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    print("\n=== DETTAGLIO PER ESEMPIO ===")
    print(df[cols].round(4).to_string(index=False))

    # (facoltativo) salva per revisione umana
    df.to_csv("ragas_results.csv", index=False)
    print("Salvato: ragas_results.csv")
    
if __name__ == "__main__":
    main()