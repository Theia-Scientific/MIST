#!/usr/bin/env python3

from pydantic import BaseModel

DEFAULT_ENABLED: bool = False
DEFAULT_ITERATIONS: int = 1
DEFAULT_SIZE: int = 3


class Erosion(BaseModel):
    enabled: bool = False
    iterations: int = 1
    size: int = 3
