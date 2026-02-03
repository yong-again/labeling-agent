"""
색상 유틸리티 - 100개 이상 객체에 대응하는 유연한 팔레트
"""

import colorsys
from typing import Tuple


def color_for_index(idx: int) -> Tuple[int, int, int]:
    """
    인덱스에 따른 고유 색상 반환 (RGB)
    HSV 기반 Golden-ratio 분포로 100개 이상에서도 시각적으로 구분 가능

    Args:
        idx: 객체 인덱스 (0부터)

    Returns:
        (R, G, B) 튜플, 0-255
    """
    # Golden ratio (1/φ) 로 hue를 분산시켜 인접 인덱스도 색상이 다르게 보이도록
    golden_ratio = 0.618033988749895
    hue = ((idx * golden_ratio) % 1.0)
    # S=0.75, V=1.0: 선명하면서도 가독성 확보
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def color_for_index_bgr(idx: int) -> Tuple[int, int, int]:
    """
    인덱스에 따른 색상 반환 (BGR - OpenCV용)

    Returns:
        (B, G, R) 튜플, 0-255
    """
    r, g, b = color_for_index(idx)
    return (b, g, r)
