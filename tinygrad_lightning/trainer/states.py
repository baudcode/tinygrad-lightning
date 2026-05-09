"""Trainer state enums (mirror lightning.pytorch states)."""
from __future__ import annotations

from enum import Enum


class TrainerFn(str, Enum):
    FITTING = "fit"
    VALIDATING = "validate"
    TESTING = "test"


class RunningStage(str, Enum):
    TRAINING = "train"
    VALIDATING = "val"
    TESTING = "test"
    SANITY_CHECKING = "sanity_check"


STAGE_TO_STR = {
    RunningStage.TRAINING: "train",
    RunningStage.VALIDATING: "val",
    RunningStage.TESTING: "test",
}
