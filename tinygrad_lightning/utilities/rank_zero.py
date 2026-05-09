"""Rank-zero logging shims.

tinygrad-lightning is single-process today, so every rank is rank zero. The
indirection exists so call sites don't change when DDP arrives.
"""
from __future__ import annotations

import logging
import warnings

_log = logging.getLogger("tinygrad_lightning")


def rank_zero_info(msg: str) -> None:
    _log.info(msg)


def rank_zero_warn(msg: str) -> None:
    warnings.warn(msg, stacklevel=2)


def rank_zero_debug(msg: str) -> None:
    _log.debug(msg)
