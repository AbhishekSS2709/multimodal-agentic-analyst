"""Query analyzer for multimodal RAG — detects whether a query targets visual or text content."""

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Individual visual keywords (each match scores +1)
_VISUAL_KEYWORDS = [
    "chart", "graph", "diagram", "image", "picture", "photo",
    "screenshot", "figure", "slide", "visualization", "plot", "map",
    "illustration", "graphic", "visual", "drawing", "sketch",
]

# Regex phrases that strongly indicate a visual query (each match scores +2)
_VISUAL_PHRASES = [
    r"show me the",
    r"what does the .* show",
    r"describe the image",
    r"in the chart",
    r"in the diagram",
    r"in the graph",
    r"in the figure",
    r"the image shows",
    r"looking at the",
    r"according to the (chart|graph|diagram|figure|image|plot)",
    r"based on the (chart|graph|diagram|figure|image|plot)",
    r"from the (chart|graph|diagram|figure|image|plot)",
]

_COMPILED_PHRASES = [re.compile(p, re.IGNORECASE) for p in _VISUAL_PHRASES]


class QueryAnalyzer:
    """Classify a search query by the modality of content it targets.

    Scoring rules
    -------------
    - Each matching visual phrase  : +2 points
    - Each matching visual keyword : +1 point

    Modality thresholds
    -------------------
    - score >= 2 -> "visual"  (text_weight=0.3, visual_weight=0.7)
    - score == 1 -> "both"    (text_weight=0.6, visual_weight=0.4)
    - score == 0 -> "text"    (text_weight=0.9, visual_weight=0.1)
    """

    def analyze(self, query: str) -> Dict:
        """Analyze *query* and return modality classification with weights.

        Parameters
        ----------
        query:
            Raw user query string.

        Returns
        -------
        dict
            Keys: ``modality`` ("visual"/"text"/"both"),
            ``text_weight`` (float), ``visual_weight`` (float),
            ``visual_score`` (int).
        """
        score = 0

        # Phrase matches (+2 each)
        for pattern in _COMPILED_PHRASES:
            if pattern.search(query):
                score += 2

        # Keyword matches (+1 each) — word-boundary aware
        query_lower = query.lower()
        for keyword in _VISUAL_KEYWORDS:
            if re.search(r"\b" + re.escape(keyword) + r"\b", query_lower):
                score += 1

        if score >= 2:
            modality = "visual"
            text_weight = 0.3
            visual_weight = 0.7
        elif score == 1:
            modality = "both"
            text_weight = 0.6
            visual_weight = 0.4
        else:
            modality = "text"
            text_weight = 0.9
            visual_weight = 0.1

        logger.debug(
            "QueryAnalyzer: query=%r score=%d modality=%s", query, score, modality
        )

        return {
            "modality": modality,
            "text_weight": text_weight,
            "visual_weight": visual_weight,
            "visual_score": score,
        }
