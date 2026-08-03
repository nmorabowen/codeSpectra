"""Optional external seismic-hazard sources for Ecuador.

NEC-SE-DS's own route to ``Z`` is the zone map (Figura 1) or the poblaciones
list (Tabla 19), and that is what governs design. This module is for the
*other* case: reading ``Z`` off a published probabilistic seismic hazard
model, to compare against the code value or to inform a site-specific study.

**No third-party data ships with codeSpectra.** The loaders read GeoJSON you
already have on disk, or fetch it from the publisher's own repository when you
explicitly ask them to. Nothing is downloaded implicitly, and nothing is
redistributed — the published layers remain the authors' work under their own
terms. :data:`PALACIOS_2023` carries the citation to reproduce alongside any
value you take from it.

Nothing here is a substitute for the code map. A value obtained this way is
tagged as non-code provenance and shows up as a note on every report it
reaches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ...core.exceptions import CodeSpectraError, InvalidInput
from ...core.reports import Report, ReportItem

if TYPE_CHECKING:
    from .nec_se_ds_2015 import NECSEDS2015

__all__ = [
    "PALACIOS_2023",
    "ContourHazardMap",
    "HazardSource",
    "HazardSourceError",
    "PGAEstimate",
    "nec_site_from_hazard",
]

#: Mean Earth radius used for the local planar approximation, km.
_EARTH_RADIUS_KM = 6371.0088


class HazardSourceError(CodeSpectraError):
    """An external hazard dataset could not be read or does not cover a site."""


@dataclass(frozen=True, slots=True)
class HazardSource:
    """Provenance and citation for an external hazard model.

    Carried on every value taken from the dataset so a report can reproduce
    the attribution without the caller having to remember it.
    """

    name: str
    authors: str
    year: int
    hazard_level: str
    url: str
    licence: str
    caveat: str = ""

    def citation(self) -> str:
        """A one-line citation to print beside any value taken from here."""
        return f"{self.authors} ({self.year}). {self.name}. {self.url}"

    def __str__(self) -> str:
        return self.citation()


#: Palacios, Celi & Poveda (2023) PSHA for continental Ecuador.
#:
#: The published layers are a mean PGA hazard map at 10% probability of
#: exceedance in 50 years — a 475-year return period, the same hazard level as
#: the NEC-SE-DS design earthquake (§4.2.1). The repository carries no licence
#: file, so it is all-rights-reserved by default: cite it, and obtain the
#: authors' permission before redistributing the layers themselves.
PALACIOS_2023 = HazardSource(
    name="SeismicHazard2023 — PSHA for continental Ecuador, mean PGA, poe 0.1 / 50 y",
    authors="Palacios, P., Celi, C., & Poveda, J. "
            "(Pontificia Universidad Catolica del Ecuador; ROSE Centre, IUSS Pavia)",
    year=2023,
    hazard_level="10% probability of exceedance in 50 years (475-year return period)",
    url="https://github.com/ppalacios92/SeismicHazard2023_poe0.1_50y",
    licence="No licence file published — all rights reserved. Cite the authors; "
            "seek their permission before redistributing the layers.",
    caveat="The authors describe this as a technical draft for academic research, "
           "subject to revision. It is not the NEC-SE-DS design map. For design "
           "in Ecuador the governing Z is NEC-SE-DS Figura 1 / Tabla 19.",
)

#: Raw URL of the PGA contour layer in the publisher's repository.
PALACIOS_2023_CONTOUR_URL = (
    "https://raw.githubusercontent.com/ppalacios92/"
    "SeismicHazard2023_poe0.1_50y/main/layers/CurvasNivelhmapmeanPGA475TR_2.js"
)


@dataclass(frozen=True, slots=True)
class PGAEstimate:
    """A PGA read off an iso-line hazard map.

    The honest product of a contour map is a **band**, not a number: the
    dataset states only where PGA crosses 0.1, 0.2, ... g. ``lower`` and
    ``upper`` are those bracketing contours, and :attr:`pga` is a linear
    interpolation between them by relative distance — convenient, but never
    more precise than the contour interval.
    """

    pga: float
    lower: float
    upper: float
    latitude: float
    longitude: float
    source: HazardSource
    distance_km: float = 0.0
    reliable: bool = True
    outside_coverage: bool = False

    @property
    def band(self) -> tuple[float, float]:
        """The bracketing contour values, in g."""
        return self.lower, self.upper

    @property
    def Z(self) -> float:
        """The estimate expressed as a NEC-SE-DS ``Z`` (PGA on rock, g)."""
        return self.pga

    def report(self) -> Report:
        """A record of the estimate, carrying the dataset's citation."""
        items = [
            ReportItem("lat", self.latitude, "Latitude", "deg"),
            ReportItem("lon", self.longitude, "Longitude", "deg"),
            ReportItem("PGA", self.pga, "Interpolated between contours", "g"),
            ReportItem("band", f"{self.lower:g} - {self.upper:g}",
                       "Bracketing contour values", "g"),
            ReportItem("d", self.distance_km, "Distance to the nearest contour",
                       "km"),
        ]
        notes = [
            f"Source: {self.source.citation()}",
            f"Hazard level: {self.source.hazard_level}",
            f"Licence: {self.source.licence}",
            self.source.caveat,
            "Read from an iso-line map: the value is no more precise than the "
            "contour interval. Use the band, not the interpolated figure, when "
            "the distinction matters.",
        ]
        if self.outside_coverage:
            notes.append(
                f"THE POINT IS OUTSIDE THE DATA FOOTPRINT (continental Ecuador "
                f"only; the Galapagos are not covered). The nearest contour is "
                f"{self.distance_km:.0f} km away, so the value below is an "
                f"extrapolation from distant geometry and carries no meaning. "
                f"Do not use it."
            )
        elif not self.reliable:
            notes.append(
                "The two nearest contours are not adjacent in value, so the "
                "point sits outside the contoured range or in a poorly "
                "constrained gap. Treat this estimate as indicative only."
            )
        return Report(
            title="External PSHA PGA estimate",
            items=tuple(items),
            notes=tuple(n for n in notes if n),
        )

    def __str__(self) -> str:
        return self.report().to_text()


class ContourHazardMap:
    """A hazard map published as iso-lines of ground acceleration.

    Loads a GeoJSON ``FeatureCollection`` of ``LineString``/``MultiLineString``
    contours whose value lives in a named property (``elev`` in the qgis2web
    exports these maps are usually published as).

    Examples
    --------
    Reading a copy you already have::

        m = ContourHazardMap.from_file("CurvasNivelhmapmeanPGA475TR_2.js")
        est = m.pga_at(lat=-0.18, lon=-78.47)   # Quito
        est.band

    Fetching from the publisher (an explicit network call, never implicit)::

        m = ContourHazardMap.from_palacios_2023(download=True)
    """

    __slots__ = ("_property", "_segments", "_source", "_values")

    def __init__(
        self,
        features: list[tuple[float, NDArray[np.float64]]],
        *,
        source: HazardSource,
        value_property: str = "elev",
    ) -> None:
        if not features:
            raise HazardSourceError("The contour dataset contains no usable lines.")
        self._segments = features
        self._values = np.array(sorted({v for v, _ in features}), dtype=float)
        self._source = source
        self._property = value_property

    # -- Loading ------------------------------------------------------------
    @classmethod
    def from_geojson(
        cls,
        data: dict[str, object],
        *,
        source: HazardSource,
        value_property: str = "elev",
    ) -> ContourHazardMap:
        """Build from an already-parsed GeoJSON ``FeatureCollection``."""
        raw = data.get("features")
        if not isinstance(raw, list):
            raise HazardSourceError(
                "Expected a GeoJSON FeatureCollection with a 'features' list."
            )
        out: list[tuple[float, NDArray[np.float64]]] = []
        for feature in raw:
            if not isinstance(feature, dict):
                continue
            props = feature.get("properties") or {}
            geom = feature.get("geometry") or {}
            if not isinstance(props, dict) or not isinstance(geom, dict):
                continue
            value = props.get(value_property)
            if not isinstance(value, (int, float)):
                continue
            gtype = geom.get("type")
            coords = geom.get("coordinates")
            if gtype == "LineString":
                parts = [coords]
            elif gtype == "MultiLineString":
                parts = coords if isinstance(coords, list) else []
            else:
                continue
            for part in parts:
                if not isinstance(part, list) or len(part) < 2:
                    continue
                arr = np.asarray(part, dtype=float)
                if arr.ndim != 2 or arr.shape[-1] < 2:
                    continue
                out.append((float(value), arr[:, :2]))
        if not out:
            raise HazardSourceError(
                f"No contour lines carrying a numeric {value_property!r} property "
                "were found in the dataset."
            )
        return cls(out, source=source, value_property=value_property)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        source: HazardSource = PALACIOS_2023,
        value_property: str = "elev",
    ) -> ContourHazardMap:
        """Load from a ``.geojson`` file, or a qgis2web ``.js`` layer.

        The ``.js`` layers wrap the GeoJSON in a ``var json_x = {...};``
        assignment; that wrapper is stripped automatically.
        """
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_geojson(
            _parse_geojson_text(text), source=source, value_property=value_property
        )

    @classmethod
    def from_palacios_2023(
        cls,
        *,
        path: str | Path | None = None,
        download: bool = False,
        url: str = PALACIOS_2023_CONTOUR_URL,
        timeout: float = 30.0,
    ) -> ContourHazardMap:
        """Load the Palacios, Celi & Poveda (2023) PGA contours.

        Parameters
        ----------
        path
            A local copy of the contour layer. Preferred: no network needed.
        download
            Fetch the layer from the publisher's repository. Must be passed
            explicitly — this module never reaches the network on its own.
        url
            Where to fetch from, if ``download`` is set.

        Notes
        -----
        codeSpectra ships none of this data. See :data:`PALACIOS_2023` for the
        citation and licence position before using or sharing any value.
        """
        if path is not None:
            return cls.from_file(path, source=PALACIOS_2023)
        if not download:
            raise HazardSourceError(
                "codeSpectra does not bundle the Palacios (2023) hazard layers. "
                "Pass path= to read a local copy, or download=True to fetch them "
                f"from {url}. Cite the authors for any value you use: "
                f"{PALACIOS_2023.citation()}"
            )
        from urllib.request import urlopen

        with urlopen(url, timeout=timeout) as response:
            text = response.read().decode("utf-8")
        return cls.from_geojson(_parse_geojson_text(text), source=PALACIOS_2023)

    # -- Introspection -------------------------------------------------------
    @property
    def source(self) -> HazardSource:
        """Provenance of the loaded dataset."""
        return self._source

    @property
    def contour_values(self) -> NDArray[np.float64]:
        """The distinct iso-line values present, ascending, in g."""
        return self._values.copy()

    @property
    def contour_interval(self) -> float:
        """Spacing between adjacent contours, in g."""
        if self._values.size < 2:
            return float("nan")
        # Rounded: contour values come from a renderer and carry float noise.
        return round(float(np.min(np.diff(self._values))), 10)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(lat_min, lat_max, lon_min, lon_max)`` of the contour data."""
        lats = np.concatenate([a[:, 1] for _, a in self._segments])
        lons = np.concatenate([a[:, 0] for _, a in self._segments])
        return (
            float(lats.min()), float(lats.max()),
            float(lons.min()), float(lons.max()),
        )

    # -- Query ---------------------------------------------------------------
    def pga_at(self, lat: float, lon: float) -> PGAEstimate:
        """Estimate PGA at a point by interpolating between nearest contours.

        Distance to each iso-line is measured on a local equirectangular
        projection, which is accurate to well under a percent over a country
        the size of Ecuador. The value is then linearly interpolated between
        the nearest contour and the nearest contour of a *different* value.
        """
        if not -90.0 <= lat <= 90.0:
            raise InvalidInput(f"Latitude must lie in [-90, 90], got {lat}.")
        if not -180.0 <= lon <= 180.0:
            raise InvalidInput(f"Longitude must lie in [-180, 180], got {lon}.")

        distances: dict[float, float] = {}
        for value, line in self._segments:
            d = _point_to_polyline_km(lat, lon, line)
            if value not in distances or d < distances[value]:
                distances[value] = d

        ordered = sorted(distances.items(), key=lambda kv: kv[1])
        v1, d1 = ordered[0]
        other = next((kv for kv in ordered[1:] if kv[0] != v1), None)

        lat_min, lat_max, lon_min, lon_max = self.bounds
        outside = not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max)

        if other is None:
            return PGAEstimate(
                pga=v1, lower=v1, upper=v1, latitude=lat, longitude=lon,
                source=self._source, distance_km=d1, reliable=False,
                outside_coverage=outside,
            )

        v2, d2 = other
        total = d1 + d2
        pga = v1 if total <= 0.0 else v1 + (v2 - v1) * (d1 / total)
        pga = float(np.clip(pga, self._values.min(), self._values.max()))

        interval = self.contour_interval
        adjacent = abs(abs(v2 - v1) - interval) < 1e-6
        return PGAEstimate(
            pga=pga,
            lower=min(v1, v2),
            upper=max(v1, v2),
            latitude=lat,
            longitude=lon,
            source=self._source,
            distance_km=d1,
            # Adjacent contours mean the point is genuinely bracketed; being
            # outside the footprint disqualifies it regardless.
            reliable=bool(adjacent) and not outside,
            outside_coverage=outside,
        )

    def __repr__(self) -> str:
        values = ", ".join(f"{v:g}" for v in self._values)
        return f"<ContourHazardMap [{values}] g | {self._source.name}>"


def nec_site_from_hazard(
    estimate: PGAEstimate,
    *,
    soil: str = "D",
    region: str = "sierra",
    occupancy: str = "otra",
    allow_unreliable: bool = False,
) -> NECSEDS2015:
    """Build a :class:`~codeSpectra.codes.nec.NECSEDS2015` site from a PSHA ``Z``.

    The resulting site carries a provenance note on every report, recording
    that ``Z`` came from an external hazard model rather than the NEC zone map.

    ``region`` still has to be supplied: NEC's ``eta`` follows provincial
    boundaries (Costa / Sierra / Oriente, with Esmeraldas and Galapagos taking
    the Sierra value), which a PGA contour map cannot tell you.
    """
    from .nec_se_ds_2015 import NECSEDS2015

    if not estimate.reliable and not allow_unreliable:
        reason = (
            "outside the data footprint"
            if estimate.outside_coverage
            else "not bracketed by adjacent contours"
        )
        raise HazardSourceError(
            f"The PGA estimate at ({estimate.latitude:.4f}, "
            f"{estimate.longitude:.4f}) is not reliable ({reason}; nearest "
            f"contour {estimate.distance_km:.0f} km away), so building a design "
            "site from it would be misleading. Supply Z from NEC-SE-DS Figura 1 "
            "or Tabla 19 instead, or pass allow_unreliable=True if you have a "
            "reason to proceed."
        )

    note = (
        f"Z = {estimate.Z:g} g was taken from an external hazard model "
        f"(band {estimate.lower:g}-{estimate.upper:g} g at "
        f"{estimate.latitude:.4f}, {estimate.longitude:.4f}), NOT from the "
        f"NEC-SE-DS zone map. {estimate.source.caveat} "
        f"Cite: {estimate.source.citation()}"
    )
    return NECSEDS2015(
        Z=estimate.Z,
        soil=soil,
        region=region,
        occupancy=occupancy,
        provenance_note=note,
    )


# -- Geometry helpers ------------------------------------------------------

def _parse_geojson_text(text: str) -> dict[str, object]:
    """Parse GeoJSON, tolerating a qgis2web ``var json_x = {...};`` wrapper."""
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        start = text.find("{")
        if start < 0:
            raise HazardSourceError("No JSON object found in the layer file.")
        text = text[start:]
    text = text.rstrip().rstrip(";")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HazardSourceError(f"Could not parse the layer as GeoJSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HazardSourceError("The layer does not contain a GeoJSON object.")
    return parsed


def _point_to_polyline_km(
    lat: float, lon: float, line: NDArray[np.float64]
) -> float:
    """Shortest distance in km from a point to a polyline of ``(lon, lat)``.

    Uses a local equirectangular projection centred on the query point, with
    longitudes scaled by ``cos(lat)``. Ecuador straddles the equator, where
    that factor is near unity, so the approximation is very tight.
    """
    scale = np.cos(np.radians(lat))
    px = np.radians(lon) * scale * _EARTH_RADIUS_KM
    py = np.radians(lat) * _EARTH_RADIUS_KM

    x = np.radians(line[:, 0]) * scale * _EARTH_RADIUS_KM
    y = np.radians(line[:, 1]) * _EARTH_RADIUS_KM

    if x.size == 1:
        return float(np.hypot(px - x[0], py - y[0]))

    ax, ay = x[:-1], y[:-1]
    bx, by = x[1:], y[1:]
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy

    # Projection parameter of the point onto each segment, clamped to it.
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(length2 > 0.0, ((px - ax) * dx + (py - ay) * dy) / length2, 0.0)
    t = np.clip(t, 0.0, 1.0)

    cx = ax + t * dx
    cy = ay + t * dy
    return float(np.min(np.hypot(px - cx, py - cy)))
