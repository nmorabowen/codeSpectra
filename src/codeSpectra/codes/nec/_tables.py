"""NEC-SE-DS 2015 site coefficient tables, transcribed from the standard.

Sources, verified against *NEC-SE-DS — Peligro Sísmico, Diseño Sismo
Resistente* (Ecuador, 2015):

  * Tabla 1  Valores del factor Z en función de la zona sísmica      (p. 27)
  * Tabla 3  Tipo de suelo y factores de sitio Fa                    (p. 31)
  * Tabla 4  Tipo de suelo y factores de sitio Fd                    (p. 31)
  * Tabla 5  Factores del comportamiento inelástico del subsuelo Fs  (p. 32)
  * Tabla 6  Tipo de uso, destino e importancia de la estructura     (p. 39)

Soil profile type F is absent from every table by design: NEC §10.5.4 and
§10.6.4 require a site-specific study, so the lookup raises rather than
returning a coefficient.
"""

from __future__ import annotations

from ...core.references import ClauseRef
from ...core.tables import InterpolatedTable

STANDARD = "NEC-SE-DS"
EDITION = "2015"

#: Tabla 1 — seismic zone to Z (fraction of g).
ZONE_Z: dict[str, float] = {
    "I": 0.15,
    "II": 0.25,
    "III": 0.30,
    "IV": 0.35,
    "V": 0.40,
    "VI": 0.50,
}

#: Tabla 1 — qualitative hazard characterisation per zone.
ZONE_HAZARD: dict[str, str] = {
    "I": "Intermedia",
    "II": "Alta",
    "III": "Alta",
    "IV": "Alta",
    "V": "Alta",
    "VI": "Muy alta",
}

#: Tabla 6 — importance coefficient by occupancy category.
IMPORTANCE: dict[str, float] = {
    "esencial": 1.5,
    "especial": 1.3,
    "otra": 1.0,
}


def _ref(clause: str, description: str, table: str | None = None) -> ClauseRef:
    return ClauseRef(
        standard=STANDARD,
        edition=EDITION,
        clause=clause,
        table=table,
        description=description,
    )


_REMEDY = (
    "Soil profile type F requires a site-specific response spectrum per "
    "NEC-SE-DS §10.5.4 / §10.6.4. Supply Fa, Fd and Fs explicitly once the "
    "study is complete."
)

_COLUMNS = (0.15, 0.25, 0.30, 0.35, 0.40, 0.50)

#: Tabla 3 — Fa, short-period soil amplification.
FA_TABLE = InterpolatedTable(
    name="Fa",
    row_label="soil type",
    col_label="Z",
    columns=_COLUMNS,
    rows={
        "A": (0.90, 0.90, 0.90, 0.90, 0.90, 0.90),
        "B": (1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
        "C": (1.40, 1.30, 1.25, 1.23, 1.20, 1.18),
        "D": (1.60, 1.40, 1.30, 1.25, 1.20, 1.12),
        "E": (1.80, 1.40, 1.25, 1.10, 1.00, 0.85),
        # Soil type F: "Vease Tabla 2 y la seccion 10.5.4" - deferred, not absent.
        "F": (None, None, None, None, None, None),
    },
    ref=_ref("3.2.2", "Coeficiente de amplificación de suelo en período corto", "3"),
    site_specific_remedy=_REMEDY,
)

#: Tabla 4 — Fd, displacement-spectrum soil amplification.
FD_TABLE = InterpolatedTable(
    name="Fd",
    row_label="soil type",
    col_label="Z",
    columns=_COLUMNS,
    rows={
        "A": (0.90, 0.90, 0.90, 0.90, 0.90, 0.90),
        "B": (1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
        "C": (1.36, 1.28, 1.19, 1.15, 1.11, 1.06),
        "D": (1.62, 1.45, 1.36, 1.28, 1.19, 1.11),
        "E": (2.10, 1.75, 1.70, 1.65, 1.60, 1.50),
        "F": (None, None, None, None, None, None),
    },
    ref=_ref("3.2.2", "Coeficiente de amplificación del espectro de desplazamientos", "4"),
    site_specific_remedy=_REMEDY,
)

#: Tabla 5 — Fs, nonlinear soil behaviour.
FS_TABLE = InterpolatedTable(
    name="Fs",
    row_label="soil type",
    col_label="Z",
    columns=_COLUMNS,
    rows={
        "A": (0.75, 0.75, 0.75, 0.75, 0.75, 0.75),
        "B": (0.75, 0.75, 0.75, 0.75, 0.75, 0.75),
        "C": (0.85, 0.94, 1.02, 1.06, 1.11, 1.23),
        "D": (1.02, 1.06, 1.11, 1.19, 1.28, 1.40),
        "E": (1.50, 1.60, 1.70, 1.80, 1.90, 2.00),
        "F": (None, None, None, None, None, None),
    },
    ref=_ref("3.2.2", "Factor de comportamiento inelástico del subsuelo", "5"),
    site_specific_remedy=_REMEDY,
)

#: §6.3.3 Método 1 — approximate period coefficients Ct and alpha.
PERIOD_COEFFICIENTS: dict[str, tuple[float, float, str]] = {
    "acero_sin_arriostramientos": (
        0.072, 0.80, "Estructuras de acero sin arriostramientos",
    ),
    "acero_con_arriostramientos": (
        0.073, 0.75, "Estructuras de acero con arriostramientos",
    ),
    "hormigon_porticos": (
        0.055, 0.90,
        "Pórticos especiales de hormigón armado sin muros estructurales ni "
        "diagonales rigidizadoras",
    ),
    "hormigon_con_muros": (
        0.055, 0.75,
        "Pórticos especiales de hormigón armado con muros estructurales o "
        "diagonales rigidizadoras, y otras estructuras basadas en muros "
        "estructurales y mampostería estructural",
    ),
}
