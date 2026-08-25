from verity.rag.semantic_retriever import SemanticRetriever


def test_semantic_retriever_returns_relevant_document():
    documents = [
        "Warehouse W3 has an inventory shortage causing product unavailability.",
        "A competitor launched a new promotion campaign.",
        "Customers reported delays because products were out of stock.",
        "The finance team approved the annual budget.",
    ]

    query = "Why are products unavailable in the warehouse?"

    retriever = SemanticRetriever()

    results = retriever.search(
        query=query,
        documents=documents,
        top_k=2,
    )

    assert len(results) == 2

    top_documents = [result.text for result in results]

    assert (
        "Warehouse W3 has an inventory shortage causing product unavailability."
        in top_documents
    )


def test_semantic_retriever_returns_empty_for_no_documents():
    retriever = SemanticRetriever()

    results = retriever.search(
        query="Why did revenue decline?",
        documents=[],
        top_k=5,
    )

    assert results == []


def test_semantic_results_are_sorted():
    documents = [
        "Annual finance budget approval.",
        "Warehouse inventory shortage caused products to be unavailable.",
        "A new competitor entered the market.",
    ]

    query = "Products are unavailable because of inventory shortage"

    retriever = SemanticRetriever()

    results = retriever.search(
        query=query,
        documents=documents,
        top_k=3,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)