from typing import List


def tokenize(text: str) -> List[str]:
    return [t for t in text.strip().lower().split() if t]
