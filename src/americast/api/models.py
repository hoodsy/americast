"""Response models: the contract a frontend builds against.

These play the same role for the API that `schemas.py` plays for the
parquet stores. A wrong column, a wrong type or a ragged array fails
here, at the boundary, rather than in a chart three layers away.

Two invariants are enforced rather than documented. Every value array
in a run's payload has exactly as many entries as `valid_times`, so a
client may index them together without checking. And every aggregation
level states whether it is validated against published data, so a
consumer cannot present an estimate as a graded forecast by accident.
"""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, model_validator

# Aggregation levels, and the one that is graded. Only the statewide
# total is compared against CAISO's published number; county and zone
# are physically-derived estimates that sum to it, and no hourly truth
# exists at those levels to check them against.
GRADED_LEVEL = "state"

Level = Literal["state", "zone", "county"]


class Plant(BaseModel):
    """One plant, as it always is. Fetch once and keep."""

    plant_id: int
    name: str
    latitude: float
    longitude: float
    capacity_mw_ac: float
    dc_capacity_mw: float
    county: str
    zone: str


class PlantList(BaseModel):
    plants: list[Plant]


class RunList(BaseModel):
    """Available model runs, newest first."""

    runs: list[datetime]


class PlantSeries(BaseModel):
    """One plant's forecast across a run's hours.

    `clearness` carries None where the sun sits below the elevation at
    which the ratio means anything. None is not zero: zero says the
    plant is making nothing under a sky that could give it something,
    None says the question does not apply yet.
    """

    plant_id: int
    mw: list[float]
    clearness: list[float | None]


class PlantFrames(BaseModel):
    run_time: datetime
    valid_times: list[datetime]
    plants: list[PlantSeries]

    @model_validator(mode="after")
    def _arrays_match_the_clock(self) -> Self:
        hours = len(self.valid_times)
        for series in self.plants:
            if len(series.mw) != hours or len(series.clearness) != hours:
                raise ValueError(
                    f"plant {series.plant_id} has a series of a different length "
                    f"than valid_times ({hours})"
                )
        return self


class LevelSeries(BaseModel):
    """One aggregation level's forecast and its clear-sky ceiling.

    `validated` is false for everything except the state total. It is
    part of the payload rather than the documentation because a
    consumer that never reads the documentation still has to know.
    """

    level: Level
    name: str
    validated: bool
    mw: list[float]
    clear_mw: list[float]

    @model_validator(mode="after")
    def _only_the_state_is_graded(self) -> Self:
        if self.validated != (self.level == GRADED_LEVEL):
            raise ValueError(
                f"{self.level} cannot be validated={self.validated}: only "
                f"{GRADED_LEVEL} is graded against published actuals"
            )
        return self


class Totals(BaseModel):
    run_time: datetime
    valid_times: list[datetime]
    levels: list[LevelSeries]

    @model_validator(mode="after")
    def _arrays_match_the_clock(self) -> Self:
        hours = len(self.valid_times)
        for series in self.levels:
            if len(series.mw) != hours or len(series.clear_mw) != hours:
                raise ValueError(
                    f"{series.level} {series.name} has a series of a different "
                    f"length than valid_times ({hours})"
                )
        return self
