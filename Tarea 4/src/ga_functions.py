import math
from typing import Tuple

F1_BOUNDS: Tuple[float, float] = (-10.0, 10.0)  
F2_BOUNDS: Tuple[float, float] = (-10.0, 10.0)  

def f1(x: float) -> float:

    if abs(x) < 1e-12:
        return 0.0
    return (math.sin(x) ** 2) / x

def f2(x: float, y: float) -> float:

    return 20.0 + x - 10.0 * math.cos(2.0 * math.pi * x) + y - 10.0 * math.cos(2.0 * math.pi * y)