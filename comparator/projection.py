from typing import Dict, List, Tuple


def _hash_sign(token: str, salt: int) -> int:
    h = hash((token, salt))
    return 1 if (h & 1) else -1


def vector_to_xy(vector: Dict[str, float]) -> Tuple[float, float]:
    # Deterministic 2D signed-hash projection for sparse vectors.
    x = 0.0
    y = 0.0
    for tok, w in vector.items():
        x += w * _hash_sign(tok, 13)
        y += w * _hash_sign(tok, 29)
    return x, y


def normalize_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    dx = (max_x - min_x) or 1.0
    dy = (max_y - min_y) or 1.0
    out = []
    for x, y in points:
        nx = (x - min_x) / dx
        ny = (y - min_y) / dy
        out.append((nx, ny))
    return out
