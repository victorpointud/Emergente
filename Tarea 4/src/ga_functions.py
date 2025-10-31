import math

def f1(x: float) -> float:
    """
    f1(x) = sin(x)^2 / x  (define f1(0) = 0 to avoid division by zero)
    Maximization.
    """
    if abs(x) < 1e-12:
        return 0.0
    return (math.sin(x) ** 2) / x

def f2(x: float, y: float) -> float:
    """
    f2(x,y) = 20 + x - 10*cos(2πx) + y - 10*cos(2πy)
    Maximization.
    """
    return 20.0 + x - 10.0 * math.cos(2.0 * math.pi * x) + y - 10.0 * math.cos(2.0 * math.pi * y)