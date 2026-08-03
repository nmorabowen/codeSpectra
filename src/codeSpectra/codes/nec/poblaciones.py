"""NEC-SE-DS Tabla 19 — poblaciones ecuatorianas y valor del factor Z.

The code's own route to ``Z``. §3.1.1 says to read it from the zone map
(Figura 1), and offers Tabla 19 as the convenient lookup — with an explicit
fallback for anywhere not listed:

    "Si se ha de disenar una estructura en una poblacion o zona que no consta
    en la lista y que se dificulte la caracterizacion de la zona en la que se
    encuentra utilizando el mapa, debe escogerse el valor de la poblacion mas
    cercana."

That sentence is what makes a coordinate lookup legitimate rather than an
invention: :meth:`Tabla19.nearest` implements the code's own nearest-town
rule. It needs coordinates, which NEC does not publish, so you supply a
gazetteer — see :class:`Gazetteer`.

All 515 entries are transcribed from the standard. Unlike the external PSHA
readers in :mod:`~codeSpectra.codes.nec.hazard`, this *is* code data, so a
value from here carries no non-code provenance warning.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.exceptions import CodeSpectraError, InvalidInput
from ...core.references import ClauseRef
from ...core.reports import Report, ReportItem

if TYPE_CHECKING:
    from .nec_se_ds_2015 import NECSEDS2015

__all__ = [
    "AmbiguousPoblacion",
    "Gazetteer",
    "Poblacion",
    "PoblacionMatch",
    "PoblacionNotFound",
    "Tabla19",
    "nec_site_from_poblacion",
    "region_for_provincia",
]

_TABLE_PATH = Path(__file__).parent / "tables" / "tabla19.json"

TABLA_19_REF = ClauseRef(
    standard="NEC-SE-DS",
    edition="2015",
    clause="10.2",
    table="19",
    description="Poblaciones ecuatorianas y valor del factor Z",
)

NEAREST_RULE_REF = ClauseRef(
    standard="NEC-SE-DS",
    edition="2015",
    clause="3.1.1",
    description="Para una poblacion que no consta en la lista, escoger el valor "
                "de la poblacion mas cercana",
)

#: Mean Earth radius, km.
_EARTH_RADIUS_KM = 6371.0088


class PoblacionNotFound(CodeSpectraError, KeyError):
    """No entry in Tabla 19 matches the requested name."""

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


class AmbiguousPoblacion(CodeSpectraError):
    """Several Tabla 19 entries share the requested name.

    Carries the candidates so the caller can disambiguate by provincia or
    canton. Note that when every candidate has the same ``Z`` the ambiguity
    does not affect the design value — :attr:`same_Z` reports that.
    """

    def __init__(self, name: str, candidates: Sequence[Poblacion]) -> None:
        self.name = name
        self.candidates = tuple(candidates)
        listing = "; ".join(
            f"{c.poblacion} ({c.canton}, {c.provincia}) Z={c.Z:g}" for c in candidates
        )
        detail = (
            " Every candidate has the same Z, so the design value is unaffected: "
            f"Z = {candidates[0].Z:g}."
            if self.same_Z
            else " The candidates differ in Z, so this must be resolved."
        )
        super().__init__(
            f"{len(candidates)} entries in NEC-SE-DS Tabla 19 are named "
            f"{name!r}: {listing}.{detail} Pass provincia= or canton= to "
            "disambiguate, or use find() to inspect them."
        )

    @property
    def same_Z(self) -> bool:
        """Whether every candidate carries the same ``Z``."""
        return len({c.Z for c in self.candidates}) == 1


def _normalise(text: str) -> str:
    """Fold to a comparable key: accents stripped, uppercased, punctuation out.

    Ecuadorian place names are written inconsistently across sources — with
    and without accents, with or without a parenthetical alternative name — so
    matching on the raw string is unreliable.
    """
    folded = unicodedata.normalize("NFKD", text)
    folded = folded.encode("ascii", "ignore").decode("ascii").upper()
    folded = re.sub(r"\(.*?\)", " ", folded)      # drop "(ALIZO)"-style aliases
    folded = re.sub(r"[^A-Z0-9 ]", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


@dataclass(frozen=True, slots=True)
class Poblacion:
    """One row of Tabla 19."""

    poblacion: str
    parroquia: str
    canton: str
    provincia: str
    Z: float

    @property
    def key(self) -> str:
        """Normalised name, for matching."""
        return _normalise(self.poblacion)

    @property
    def zone(self) -> str | None:
        """Seismic zone I-VI implied by ``Z``, or None if it is non-standard."""
        for zone, value in _ZONE_Z.items():
            if abs(self.Z - value) < 1e-9:
                return zone
        return None

    def __str__(self) -> str:
        return f"{self.poblacion} ({self.canton}, {self.provincia}) Z={self.Z:g}"


_ZONE_Z = {"I": 0.15, "II": 0.25, "III": 0.30, "IV": 0.35, "V": 0.40, "VI": 0.50}


# -- Region (eta) by province ----------------------------------------------
#
# NEC-SE-DS §3.3.1 sets eta by region: 1.80 for the Costa provinces except
# Esmeraldas, 2.48 for the Sierra provinces plus Esmeraldas and Galapagos,
# 2.60 for the Oriente. Since Tabla 19 names the provincia of every entry,
# the region follows for all but the genuinely unclear cases below.

_COSTA = {"GUAYAS", "MANABI", "LOS RIOS", "EL ORO", "SANTA ELENA"}
_SIERRA = {
    "AZUAY", "BOLIVAR", "CANAR", "CARCHI", "CHIMBORAZO", "COTOPAXI",
    "IMBABURA", "LOJA", "PICHINCHA", "TUNGURAHUA",
    "ESMERALDAS",   # named explicitly in §3.3.1 despite being coastal
    "GALAPAGOS",    # named explicitly in §3.3.1
}
_ORIENTE = {
    "MORONA SANTIAGO", "NAPO", "ORELLANA", "PASTAZA",
    "SUCUMBIOS", "ZAMORA CHINCHIPE",
}

#: Provinces §3.3.1 does not place in any of its three groups.
AMBIGUOUS_PROVINCES = {
    "STO DOMINGO DE LOS TSACHILAS": (
        "Santo Domingo de los Tsachilas was created in 2007, after the region "
        "groupings §3.3.1 inherits, and sits on the Costa-Sierra transition. "
        "NEC-SE-DS does not assign it."
    ),
    "ZONA NO DELIMITADA": (
        "'Zona no delimitada' is not a province and has no region assignment."
    ),
}


def region_for_provincia(provincia: str) -> str:
    """Region controlling ``eta`` for an Ecuadorian province (§3.3.1).

    Returns ``"costa"``, ``"sierra"`` or ``"oriente"``.

    Raises
    ------
    InvalidInput
        The province is one §3.3.1 does not place in a region — currently
        Santo Domingo de los Tsachilas and 'Zona no delimitada'. Choose the
        region explicitly for those.
    """
    key = _normalise(provincia)
    if key in _COSTA:
        return "costa"
    if key in _SIERRA:
        return "sierra"
    if key in _ORIENTE:
        return "oriente"
    if key in AMBIGUOUS_PROVINCES:
        raise InvalidInput(
            f"NEC-SE-DS §3.3.1 does not assign a region to {provincia!r}. "
            f"{AMBIGUOUS_PROVINCES[key]} Supply region= explicitly "
            "('costa', 'sierra' or 'oriente')."
        )
    raise InvalidInput(
        f"Unknown Ecuadorian province {provincia!r}. Supply region= explicitly."
    )


@dataclass(frozen=True, slots=True)
class PoblacionMatch:
    """A Tabla 19 entry located by proximity, with the distance to it."""

    poblacion: Poblacion
    distance_km: float
    latitude: float
    longitude: float

    @property
    def Z(self) -> float:
        """Design PGA on rock for the matched poblacion, in g."""
        return self.poblacion.Z

    def report(self) -> Report:
        """A record of the nearest-town lookup, with its citations."""
        p = self.poblacion
        return Report(
            title="NEC-SE-DS Tabla 19 nearest-poblacion lookup",
            items=(
                ReportItem("lat", self.latitude, "Query latitude", "deg"),
                ReportItem("lon", self.longitude, "Query longitude", "deg"),
                ReportItem("Poblacion", p.poblacion, "Nearest listed poblacion",
                           "", NEAREST_RULE_REF),
                ReportItem("Canton", p.canton, "Canton"),
                ReportItem("Provincia", p.provincia, "Provincia"),
                ReportItem("d", self.distance_km, "Distance to that poblacion", "km"),
                ReportItem("Z", p.Z, "Design PGA on rock", "g", TABLA_19_REF),
            ),
            notes=(
                "NEC-SE-DS §3.1.1 permits taking Z from the nearest listed "
                "poblacion where a site is not itself listed. Confirm against "
                "Figura 1 that no zone boundary lies between the two.",
            ),
        )

    def __str__(self) -> str:
        return self.report().to_text()


class Gazetteer:
    """Coordinates for place names, used by :meth:`Tabla19.nearest`.

    NEC publishes no coordinates for Tabla 19, so they have to come from
    somewhere else. Any source works as long as it can be expressed as
    ``name -> (lat, lon)``; names are matched on the same normalised key the
    table uses, so accents and parenthetical aliases do not matter.

    Examples
    --------
    >>> g = Gazetteer({"Quito": (-0.1807, -78.4678),
    ...                "Guayaquil": (-2.1709, -79.9224)})
    >>> len(g)
    2
    """

    __slots__ = ("_points",)

    def __init__(self, points: Mapping[str, tuple[float, float]]) -> None:
        resolved: dict[str, tuple[float, float]] = {}
        for name, (lat, lon) in points.items():
            if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
                raise InvalidInput(
                    f"Coordinates for {name!r} are out of range: ({lat}, {lon})."
                )
            resolved[_normalise(name)] = (float(lat), float(lon))
        if not resolved:
            raise InvalidInput("A gazetteer needs at least one place.")
        self._points = resolved

    @classmethod
    def from_geojson_points(
        cls,
        data: Mapping[str, object],
        *,
        name_property: str = "id",
    ) -> Gazetteer:
        """Build from a GeoJSON ``Point``/``MultiPoint`` FeatureCollection.

        Convenient for the city layers published alongside hazard maps.
        """
        features = data.get("features")
        if not isinstance(features, list):
            raise InvalidInput("Expected a GeoJSON FeatureCollection.")
        points: dict[str, tuple[float, float]] = {}
        for feature in features:
            if not isinstance(feature, dict):
                continue
            props = feature.get("properties") or {}
            geom = feature.get("geometry") or {}
            if not isinstance(props, dict) or not isinstance(geom, dict):
                continue
            name = props.get(name_property)
            coords = geom.get("coordinates")
            if not isinstance(name, str):
                continue
            if geom.get("type") == "MultiPoint" and isinstance(coords, list) and coords:
                coords = coords[0]
            if (
                geom.get("type") in ("Point", "MultiPoint")
                and isinstance(coords, list)
                and len(coords) >= 2
            ):
                points[name] = (float(coords[1]), float(coords[0]))
        if not points:
            raise InvalidInput(
                f"No point features carrying a {name_property!r} property were found."
            )
        return cls(points)

    @classmethod
    def from_file(cls, path: str | Path, *, name_property: str = "id") -> Gazetteer:
        """Load point coordinates from a ``.geojson`` or qgis2web ``.js`` layer."""
        text = Path(path).read_text(encoding="utf-8")
        start = text.find("{")
        if start < 0:
            raise InvalidInput("No JSON object found in the layer file.")
        payload = json.loads(text[start:].rstrip().rstrip(";"))
        return cls.from_geojson_points(payload, name_property=name_property)

    def get(self, name: str) -> tuple[float, float] | None:
        """Coordinates for ``name``, or None if it is not in the gazetteer."""
        return self._points.get(_normalise(name))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and _normalise(name) in self._points

    def __len__(self) -> int:
        return len(self._points)

    def __repr__(self) -> str:
        return f"<Gazetteer {len(self._points)} places>"


class Tabla19(Sequence[Poblacion]):
    """NEC-SE-DS Tabla 19, the poblacion-to-``Z`` list.

    Examples
    --------
    >>> t = Tabla19.load()
    >>> len(t)
    515
    >>> t.by_name("Quito").Z
    0.4
    >>> t.by_name("cuenca").Z          # matching ignores case and accents
    0.25
    """

    __slots__ = ("_index", "_rows")

    def __init__(self, rows: Iterable[Poblacion]) -> None:
        self._rows = tuple(rows)
        if not self._rows:
            raise InvalidInput("Tabla 19 cannot be empty.")
        index: dict[str, list[Poblacion]] = {}
        for row in self._rows:
            index.setdefault(row.key, []).append(row)
        self._index = {k: tuple(v) for k, v in index.items()}

    @classmethod
    def load(cls) -> Tabla19:
        """The table as shipped with codeSpectra (cached)."""
        return _load_table()

    # -- Sequence protocol --------------------------------------------------
    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Poblacion:  # type: ignore[override]
        return self._rows[index]

    def __iter__(self) -> Iterator[Poblacion]:
        return iter(self._rows)

    # -- Lookup by name -----------------------------------------------------
    def find(self, name: str) -> tuple[Poblacion, ...]:
        """Every entry whose name matches, ignoring case and accents."""
        return self._index.get(_normalise(name), ())

    def search(self, text: str) -> tuple[Poblacion, ...]:
        """Every entry whose name contains ``text`` as a substring."""
        needle = _normalise(text)
        if not needle:
            return ()
        return tuple(r for r in self._rows if needle in r.key)

    def by_name(
        self,
        name: str,
        *,
        provincia: str | None = None,
        canton: str | None = None,
    ) -> Poblacion:
        """The single entry named ``name``.

        Raises
        ------
        PoblacionNotFound
            Nothing matches. The message suggests close names where it can.
        AmbiguousPoblacion
            Several entries share the name and the filters did not narrow it
            to one. Thirty names in Tabla 19 are duplicated across provinces.
        """
        candidates = self.find(name)
        if not candidates:
            hint = ""
            near = self.search(name.split()[0]) if name.split() else ()
            if near:
                names = sorted({c.poblacion for c in near})[:6]
                hint = f" Did you mean: {', '.join(names)}?"
            raise PoblacionNotFound(
                f"{name!r} is not listed in NEC-SE-DS Tabla 19.{hint} "
                "Per §3.1.1 you may instead take Z from the nearest listed "
                "poblacion — see Tabla19.nearest()."
            )
        if provincia is not None:
            key = _normalise(provincia)
            candidates = tuple(c for c in candidates if _normalise(c.provincia) == key)
        if canton is not None:
            key = _normalise(canton)
            candidates = tuple(c for c in candidates if _normalise(c.canton) == key)
        if not candidates:
            raise PoblacionNotFound(
                f"No Tabla 19 entry named {name!r} matches the given "
                "provincia/canton filters."
            )
        if len(candidates) > 1:
            raise AmbiguousPoblacion(name, candidates)
        return candidates[0]

    # -- Lookup by position --------------------------------------------------
    def nearest(
        self,
        lat: float,
        lon: float,
        gazetteer: Gazetteer,
        *,
        max_distance_km: float | None = None,
    ) -> PoblacionMatch:
        """The listed poblacion nearest to a point — the §3.1.1 rule.

        Parameters
        ----------
        gazetteer
            Coordinates for the poblacion names. Only entries the gazetteer
            knows can be matched, so its coverage bounds the result.
        max_distance_km
            Refuse to return a match further away than this. There is no code
            limit, but a town hundreds of km away is not "la poblacion mas
            cercana" in any useful sense, and a zone boundary will usually lie
            between.

        Raises
        ------
        PoblacionNotFound
            The gazetteer covers none of the table, or the nearest covered
            entry is beyond ``max_distance_km``.
        """
        if not -90.0 <= lat <= 90.0:
            raise InvalidInput(f"Latitude must lie in [-90, 90], got {lat}.")
        if not -180.0 <= lon <= 180.0:
            raise InvalidInput(f"Longitude must lie in [-180, 180], got {lon}.")

        best: tuple[float, Poblacion] | None = None
        for row in self._rows:
            point = gazetteer.get(row.poblacion)
            if point is None:
                continue
            d = _haversine_km(lat, lon, point[0], point[1])
            if best is None or d < best[0]:
                best = (d, row)

        if best is None:
            raise PoblacionNotFound(
                f"The gazetteer ({len(gazetteer)} places) covers none of the "
                f"{len(self._rows)} poblaciones in Tabla 19, so no nearest "
                "town could be found. Check that its place names match."
            )
        distance, row = best
        if max_distance_km is not None and distance > max_distance_km:
            raise PoblacionNotFound(
                f"The nearest covered poblacion, {row.poblacion}, is "
                f"{distance:.0f} km away, beyond the {max_distance_km:g} km "
                "limit. Widen the gazetteer, raise max_distance_km, or read Z "
                "from Figura 1 directly."
            )
        point = gazetteer.get(row.poblacion)
        assert point is not None
        return PoblacionMatch(
            poblacion=row, distance_km=distance,
            latitude=lat, longitude=lon,
        )

    def covered_by(self, gazetteer: Gazetteer) -> tuple[Poblacion, ...]:
        """Entries the gazetteer can place — use to check its coverage."""
        return tuple(r for r in self._rows if gazetteer.get(r.poblacion) is not None)

    # -- Summary --------------------------------------------------------------
    @property
    def provincias(self) -> tuple[str, ...]:
        """Distinct provinces present, sorted."""
        return tuple(sorted({r.provincia for r in self._rows}))

    def by_provincia(self, provincia: str) -> tuple[Poblacion, ...]:
        """Every entry in a province."""
        key = _normalise(provincia)
        return tuple(r for r in self._rows if _normalise(r.provincia) == key)

    def __repr__(self) -> str:
        return f"<Tabla19 {len(self._rows)} poblaciones, {len(self.provincias)} provincias>"


@lru_cache(maxsize=1)
def _load_table() -> Tabla19:
    payload = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
    rows = [
        Poblacion(poblacion=r[0], parroquia=r[1], canton=r[2],
                  provincia=r[3], Z=float(r[4]))
        for r in payload["rows"]
    ]
    return Tabla19(rows)


def nec_site_from_poblacion(
    poblacion: Poblacion | PoblacionMatch,
    *,
    soil: str = "D",
    region: str | None = None,
    occupancy: str = "otra",
) -> NECSEDS2015:
    """Build a NEC site from a Tabla 19 entry.

    ``region`` is derived from the province via :func:`region_for_provincia`
    unless given. That derivation is the §3.3.1 grouping, so it is code-based
    rather than a guess — but two provinces the standard does not place
    (Santo Domingo de los Tsachilas, 'Zona no delimitada') will raise, and
    need ``region`` supplied.
    """
    from .nec_se_ds_2015 import NECSEDS2015

    row = poblacion.poblacion if isinstance(poblacion, PoblacionMatch) else poblacion
    resolved = region if region is not None else region_for_provincia(row.provincia)

    note = f"Z = {row.Z:g} g from NEC-SE-DS Tabla 19: {row}."
    if isinstance(poblacion, PoblacionMatch):
        note += (
            f" Located as the nearest listed poblacion to "
            f"({poblacion.latitude:.4f}, {poblacion.longitude:.4f}), "
            f"{poblacion.distance_km:.1f} km away, per §3.1.1. Confirm against "
            "Figura 1 that no zone boundary lies between the two."
        )
    if region is None:
        note += f" Region '{resolved}' derived from provincia per §3.3.1."

    return NECSEDS2015(
        Z=row.Z,
        soil=soil,
        region=resolved,
        occupancy=occupancy,
        provenance_note=note,
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    from math import asin, cos, radians, sin, sqrt

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2.0) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * asin(sqrt(a))
