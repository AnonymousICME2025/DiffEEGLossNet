"""
BCICIV shape (batch_size, 25, 448), 4837 original samples, 4837 generated samples
"""

# from eval import *  # import this module like this please

from .inception_score import compute_is
from .frechet_inception_distance import compute_fid
