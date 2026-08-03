"""NEC — Norma Ecuatoriana de la Construcción (Ecuador)."""

from .hazard import (
    PALACIOS_2023,
    ContourHazardMap,
    HazardSource,
    HazardSourceError,
    PGAEstimate,
    nec_site_from_hazard,
)
from .nec_se_ds_2015 import (
    NECSEDS2015,
    OccupancyCategory,
    Region,
    SeismicZone,
    SoilType,
)
from .poblaciones import (
    AmbiguousPoblacion,
    Gazetteer,
    Poblacion,
    PoblacionMatch,
    PoblacionNotFound,
    Tabla19,
    nec_site_from_poblacion,
    region_for_provincia,
)

__all__ = [
    "NECSEDS2015",
    "PALACIOS_2023",
    "AmbiguousPoblacion",
    "ContourHazardMap",
    "Gazetteer",
    "HazardSource",
    "HazardSourceError",
    "OccupancyCategory",
    "PGAEstimate",
    "Poblacion",
    "PoblacionMatch",
    "PoblacionNotFound",
    "Region",
    "SeismicZone",
    "SoilType",
    "Tabla19",
    "nec_site_from_hazard",
    "nec_site_from_poblacion",
    "region_for_provincia",
]
