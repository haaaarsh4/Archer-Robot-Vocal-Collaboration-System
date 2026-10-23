"""Temporary: row profile of the home caption, to line our render up with the
reference frame. Delete after use."""
import sys

import numpy as np
from PIL import Image

path = sys.argv[1]
g = np.array(Image.open(path).convert("L")).astype(float)
bg = np.median(g[600:790, 5:60])
print("paper", bg)
for y in range(int(sys.argv[2]) if len(sys.argv) > 2 else 600,
               int(sys.argv[3]) if len(sys.argv) > 3 else 745, 3):
    band = g[y:y + 3, 500:960]
    print(y, int((band < bg - 60).sum()))
