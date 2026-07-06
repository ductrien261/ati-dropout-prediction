"""
Radar chart baseline: vẽ đa giác 20 đỉnh (1 đỉnh/activity, bán kính theo tỷ trọng click) lên canvas cố định bằng PIL
"""
import numpy as np
from PIL import Image, ImageDraw

from src.ati.transforms.common import N_ACTIVITIES, is_sparse_week, build_weekly_vector

CANVAS_SIZE = 64


def build_radar_image(week_daily: dict, n: int = N_ACTIVITIES,
                       canvas_size: int = CANVAS_SIZE) -> np.ndarray:
    """Trả về ảnh grayscale (canvas_size, canvas_size), giá trị [0,1]."""
    if not week_daily or is_sparse_week(week_daily):
        return np.zeros((canvas_size, canvas_size), dtype=np.float32)

    vec = build_weekly_vector(week_daily, n)
    total = vec.sum()
    normalized = vec / total if total > 0 else vec

    center = canvas_size / 2
    max_radius = canvas_size / 2 - 2
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    points = []
    for angle, value in zip(angles, normalized):
        r = value * max_radius
        x = center + r * np.cos(angle)
        y = center - r * np.sin(angle)
        points.append((x, y))

    img = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)
    if len(points) >= 3:
        draw.polygon(points, fill=255, outline=255)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr