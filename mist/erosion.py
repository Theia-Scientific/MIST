#!/usr/bin/env python3

from pydantic import BaseModel

DEFAULT_ENABLED: bool = False
DEFAULT_ITERATIONS: int = 1
DEFAULT_SIZE: int = 3


class Configuration(BaseModel):
    enabled: bool = DEFAULT_ENABLED
    iterations: int = DEFAULT_ITERATIONS
    size: int = DEFAULT_SIZE
