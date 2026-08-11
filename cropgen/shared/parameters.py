from typing import Callable
import numpy as np


class Parameter:
    """
    Represents a probability distribution or a single value.
    Useful for transformation configuration.
    """

    __slots__ = ("_value",)

    def __init__(self, value: float | Callable[[], float]):
        self._value = value

    def __call__(self) -> float:
        if isinstance(self._value, float):
            return self._value

        elif callable(self._value):
            return self._value()  # ty: ignore[call-top-callable, invalid-return-type]

        else:
            raise ValueError("Value is neither a callable nor a float.")


class NormalDistribution(Parameter):
    __slots__ = ("_mean", "_sigma")

    def __init__(self, mean: float = 0, sigma: float = 1):
        self._mean = mean
        self._sigma = sigma

    def __call__(self) -> float:
        return float(np.random.normal(self._mean, self._sigma))


class UniformDistribution(Parameter):
    __slots__ = ("_min", "_max")

    def __init__(self, low: float, high: float):
        self._min = low
        self._max = high

    def __call__(self) -> float:
        return float(np.random.uniform(self._min, self._max))
