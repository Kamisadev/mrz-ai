"""Measuring the recognizer against real passports.

Kept out of `inference` and out of `training` on purpose: it needs both, plus
`serve.crop` to frame the lines, and it is the one place in the project where a
number comes from something the generator did not draw.
"""

from .real import RealDocument, RealResult, load_real_set, measure_real

__all__ = ["RealDocument", "RealResult", "load_real_set", "measure_real"]
