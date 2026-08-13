from typing import Callable
import numpy as np


class Parameter:
    """
    Represents a probability distribution or a single value.
    Useful for transformation configuration.
    """

    __slots__ = ("_value",)

    # TODO: solve compatibility with NormalDistribution and UniformDistribution.

    def __init__(self, value: "Parameter | float | Callable[[], float]"):
        if isinstance(value, Parameter):
            self._value = value._value
        else:
            self._value = value

    def __call__(self) -> float:
        if isinstance(self._value, (float, int)):
            return self._value

        elif callable(self._value):
            return self._value()  # ty: ignore[call-top-callable, invalid-return-type]

        else:
            raise ValueError(
                f"Value is neither a callable nor a float or int, but {self._value}"
            )


class NormalDistribution(Parameter):
    __slots__ = ("_mean", "_sigma")

    def __init__(self, mean: float = 0, sigma: float = 1):
        self._mean = mean
        self._sigma = sigma

    def __call__(self) -> float:
        return float(np.random.normal(self._mean, self._sigma))

    def __repr__(self):
        return f"<N({self._mean},{self._sigma})>"


class UniformDistribution(Parameter):
    __slots__ = ("_min", "_max")

    def __init__(self, low: float, high: float):
        self._min = low
        self._max = high

    def __call__(self) -> float:
        return float(np.random.uniform(self._min, self._max))

    def __repr__(self):
        return f"<U({self._min},{self._max})>"


def instanciate_if_parameter(value: Parameter | float) -> float:
    if isinstance(value, Parameter):
        return value()
    else:
        return value
