from dataclasses import dataclass
from typing import Optional
from sentence_transformers import CrossEncoder
from src.ingestion.chunking import Chunk


@dataclass
class RankedChunk:
    chunk: Chunk
    retrieval_score: float
    rerank_score: Optional[float] = None


@dataclass
class RerankResult:
    chunks: list[RankedChunk]
    applied: bool
    model_status: str


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None
        self._load_error: Optional[str] = None
        self._warned = False

    @property
    def status(self) -> str:
        if self._load_error is not None:
            return "unavailable"
        if self._model is not None:
            return "ready"
        return "not_loaded"

    def _get_model(self) -> Optional[CrossEncoder]:
        if self._load_error is not None:
            return None

        if self._model is None:
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                self._load_error = str(exc)
                if not self._warned:
                    print(
                        "Reranker unavailable, falling back to retrieval order:"
                        f" {exc}"
                    )
                    self._warned = True
                return None

        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_n: Optional[int] = None,
    ) -> RerankResult:
        if not candidates:
            return RerankResult(
                chunks=[],
                applied=False,
                model_status=self.status,
            )

        ranked_by_retrieval = [
            RankedChunk(chunk=chunk, retrieval_score=score)
            for chunk, score in sorted(
                candidates,
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        model = self._get_model()
        if model is None:
            limit = len(ranked_by_retrieval) if top_n is None else top_n
            return RerankResult(
                chunks=ranked_by_retrieval[:limit],
                applied=False,
                model_status=self.status,
            )

        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = model.predict(pairs)

        reranked = sorted(
            [
                RankedChunk(
                    chunk=chunk,
                    retrieval_score=retrieval_score,
                    rerank_score=float(score),
                )
                for (chunk, retrieval_score), score in zip(candidates, scores)
            ],
            key=lambda item: item.rerank_score if item.rerank_score is not None else 0.0,
            reverse=True,
        )

        limit = len(reranked) if top_n is None else top_n

        return RerankResult(
            chunks=reranked[:limit],
            applied=True,
            model_status=self.status,
        )
