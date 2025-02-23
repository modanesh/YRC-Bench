import random
import numpy as np
from scipy.stats import sem


def area_under_curve(points):
    """
    Computes the area under the curve using the trapezoidal rule.

    Args:
        points (list of tuples): A list of (x, y) data points sorted by x in ascending order.

    Returns:
        float: The computed area under the curve.
    """
    # Ensure the points are sorted by x
    points = sorted(points, key=lambda p: p[0])

    # Initialize area
    area = 0.0

    # Loop through consecutive points
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]

        # Compute the trapezoid area
        trapezoid_area = 0.5 * (x2 - x1) * (y1 + y2)
        area += trapezoid_area

    return area


def compute_metric(arr):

    N = 1000
    K = 256 if len(arr[0][1]) > 256 else 32
    samples = []
    for _ in range(N):
        points = []
        for x, pool in arr:
            sample = random.sample(pool, K)
            points.append((x, np.mean(sample)))
        samples.append(area_under_curve(points))

    return np.mean(samples), np.std(samples)

