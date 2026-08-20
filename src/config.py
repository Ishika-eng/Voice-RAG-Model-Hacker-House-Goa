import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    generation_model: str = os.getenv("GENERATION_MODEL", "openai/gpt-oss-120b")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    qdrant_path: str = os.getenv("QDRANT_PATH", "./data/qdrant_local")
    top_k: int = int(os.getenv("TOP_K", "5"))

    hf_dataset: str = os.getenv("HF_DATASET", "ai4bharat/MSMARCO-XI")
    hf_dataset_lang: str = os.getenv("HF_DATASET_LANG", "hi")
    ingest_limit: int = int(os.getenv("INGEST_LIMIT", "20000"))

    offtopic_sim_threshold: float = float(os.getenv("OFFTOPIC_SIM_THRESHOLD", "0.72"))
    groundedness_overlap_threshold: float = float(
        os.getenv("GROUNDEDNESS_OVERLAP_THRESHOLD", "0.20")
    )


settings = Settings()
