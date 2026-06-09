from .base import BaseReranker, register_reranker
from ..protocol import Paper, CorpusPaper
from openai import OpenAI
from loguru import logger
import numpy as np
import json
import re


@register_reranker("llm")
class LlmReranker(BaseReranker):
    """Rerank candidates by asking an LLM to score their relevance to the user's
    interest profile (the most recently added papers in the Zotero library)."""

    def _build_profile(self, corpus: list[CorpusPaper]) -> str:
        corpus = sorted(corpus, key=lambda x: x.added_date, reverse=True)
        profile_size = self.config.reranker.llm.get("profile_size") or 50
        titles = [c.title for c in corpus[:profile_size]]
        return "\n".join(f"- {t}" for t in titles)

    def _score_batch(self, client: OpenAI, profile: str, batch: list[Paper]) -> dict[int, float]:
        papers_text = "\n\n".join(
            f"[{i}] Title: {p.title}\nAbstract: {p.abstract[:1500]}"
            for i, p in enumerate(batch)
        )
        prompt = (
            "A researcher's interests are represented by the recent papers in their "
            f"library (most recent first, higher importance):\n{profile}\n\n"
            "Score each of the following new papers from 0 (irrelevant) to 10 "
            "(highly relevant) by how well it matches the researcher's interests. "
            "Return ONLY a JSON array of objects {\"index\": <int>, \"score\": <float>}, "
            "one per paper, with no other text.\n\nNew papers:\n" + papers_text
        )
        response = client.chat.completions.create(
            model=self.config.reranker.llm.model,
            max_tokens=self.config.reranker.llm.get("max_tokens") or 4096,
            messages=[
                {"role": "system", "content": "You are a research assistant who scores how relevant new papers are to a researcher's interests."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        match = re.search(r"\[.*\]", content, re.DOTALL)
        scores = json.loads(match.group(0))
        return {int(s["index"]): float(s["score"]) for s in scores}

    def rerank(self, candidates: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        client = OpenAI(api_key=self.config.reranker.llm.key, base_url=self.config.reranker.llm.base_url)
        profile = self._build_profile(corpus)
        batch_size = self.config.reranker.llm.get("batch_size") or 20
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            try:
                scores = self._score_batch(client, profile, batch)
            except Exception as e:
                logger.warning(f"LLM rerank failed for batch starting at {start}: {e}")
                scores = {}
            for i, p in enumerate(batch):
                p.score = scores.get(i, 0.0)
        return sorted(candidates, key=lambda x: x.score, reverse=True)

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        raise NotImplementedError("LlmReranker scores candidates directly via rerank().")
