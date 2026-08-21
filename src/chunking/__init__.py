from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    strategy: str
    doc_id: str
    metadata: dict = field(default_factory=dict)