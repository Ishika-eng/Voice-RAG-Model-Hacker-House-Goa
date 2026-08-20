#!/usr/bin/env python3
"""Offline smoke test: exercises chunking -> embedding -> Qdrant ingest ->
retrieval -> guardrails against a few rows shaped exactly like the real
MSMARCO-XI parquet schema (source_lang/target_lang/meta/Answer/query_id/
query_type/passages{English_passages,Translated_passages,is_selected}/
Eng_Query/Eng_Answer/query), confirmed via the dataset's parquet footer.
Doesn't require downloading the full dataset -- useful when the network
can't sustain the multi-GB HF download.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import build_chunks_for_row
from src.embeddings import embed_passages
from src.vector_store import upsert_chunks
from src.retrieval import retrieve, max_similarity
from src.guardrails import check_input_safety, check_on_topic, check_groundedness

SAMPLE_ROWS = [
    {
        "query_id": 1001,
        "query_type": "DESCRIPTION",
        "query": "भारत की राजधानी क्या है",
        "Answer": "भारत की राजधानी नई दिल्ली है।",
        "passages": {
            "English_passages": [
                "New Delhi is the capital of India. It is located in the northern part of the country.",
                "Mumbai is the financial capital of India, known for Bollywood and its busy port.",
            ],
            "Translated_passages": [
                "नई दिल्ली भारत की राजधानी है। यह देश के उत्तरी भाग में स्थित है। दिल्ली में कई ऐतिहासिक स्मारक हैं जैसे लाल किला और कुतुब मीनार। यह भारत सरकार की सीट भी है।",
                "मुंबई भारत की आर्थिक राजधानी है, जो बॉलीवुड और अपने व्यस्त बंदरगाह के लिए जानी जाती है। यह महाराष्ट्र राज्य में स्थित है।",
            ],
            "is_selected": [1, 0],
        },
    },
    {
        "query_id": 1002,
        "query_type": "NUMERIC",
        "query": "ताजमहल कब बनाया गया था",
        "Answer": "ताजमहल 1632 में बनाया जाना शुरू हुआ था।",
        "passages": {
            "English_passages": [
                "The Taj Mahal was built by Mughal emperor Shah Jahan starting in 1632, completed around 1653.",
            ],
            "Translated_passages": [
                "ताजमहल का निर्माण मुगल सम्राट शाहजहाँ ने 1632 में शुरू करवाया था और यह लगभग 1653 में पूरा हुआ। यह आगरा में स्थित है और यूनेस्को विश्व धरोहर स्थल है। इसे अपनी पत्नी मुमताज़ महल की याद में बनवाया गया था।",
            ],
            "is_selected": [1],
        },
    },
]


def main():
    print("=== 1. Chunking ===")
    all_chunks_by_strategy = {"fixed_size": [], "passage_native": [], "semantic": []}
    for row in SAMPLE_ROWS:
        chunks = build_chunks_for_row(row)
        for c in chunks:
            all_chunks_by_strategy[c.strategy].append(c)
    for strategy, chunks in all_chunks_by_strategy.items():
        print(f"  {strategy}: {len(chunks)} chunks")
        for c in chunks[:2]:
            print(f"    - [{c.metadata.get('is_selected')}] {c.text[:80]}...")

    print("\n=== 2. Embedding + Qdrant upsert ===")
    for strategy, chunks in all_chunks_by_strategy.items():
        if not chunks:
            continue
        vectors = embed_passages([c.text for c in chunks])
        upsert_chunks(strategy, chunks, vectors)
        print(f"  upserted {len(chunks)} into collection for '{strategy}'")

    print("\n=== 3. Retrieval (on-topic query) ===")
    query = "दिल्ली भारत की राजधानी कहाँ है"
    results = retrieve(query, top_k=3)
    for r in results:
        print(f"  [{r.strategy}] score={r.score:.3f} {r.text[:80]}...")

    print("\n=== 4. Guardrails ===")
    print("  input_safety(query):", check_input_safety(query))
    print("  on_topic(retrieved):", check_on_topic(results))
    offtopic_results = retrieve("What is the boiling point of nitrogen on Mars", top_k=3)
    print("  on_topic(off-topic query):", check_on_topic(offtopic_results), f"(best sim={max([r.score for r in offtopic_results], default=0):.3f})")

    fake_grounded_answer = "नई दिल्ली भारत की राजधानी है।"
    fake_ungrounded_answer = "The Eiffel Tower is located in Paris and was built in 1889."
    print("  groundedness(grounded answer):", check_groundedness(fake_grounded_answer, results))
    print("  groundedness(ungrounded answer):", check_groundedness(fake_ungrounded_answer, results))

    print("\nSmoke test complete -- chunking, embedding, vector store, retrieval,")
    print("and all three guardrail checks executed successfully end to end.")


if __name__ == "__main__":
    main()
