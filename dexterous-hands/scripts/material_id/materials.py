"""Material table for the fingertip material-discrimination experiment.

Friction coefficients are for a dry silicone/rubber fingertip sliding on the
surface; thermal properties are handbook values for the bulk material. The set is
chosen so neither cue separates every pair on its own: aluminum and stainless
steel are thermally indistinguishable through a low-conductivity fingertip skin,
oak and ABS have nearly identical conductivity, and silicone rubber is thermally
close to both but far away in friction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """One plate surface: how it grips and how it conducts heat.

    Parameters
    ----------
    name : str
        Class label used in the dataset and in plots.
    friction : float
        Coulomb friction coefficient against the silicone fingertip, dry.
    conductivity : float
        Thermal conductivity in W/(m*K).
    density : float
        Density in kg/m^3.
    specific_heat : float
        Specific heat capacity in J/(kg*K).
    emissivity : float
        Surface emissivity, 0 to 1.
    """

    name: str
    friction: float
    conductivity: float
    density: float
    specific_heat: float
    emissivity: float


MATERIALS: tuple[Material, ...] = (
    Material("aluminum", friction=0.45, conductivity=205.0, density=2700.0, specific_heat=900.0, emissivity=0.09),
    Material("steel", friction=0.35, conductivity=16.0, density=8000.0, specific_heat=500.0, emissivity=0.28),
    Material("glass", friction=0.70, conductivity=1.05, density=2500.0, specific_heat=840.0, emissivity=0.92),
    Material("ceramic", friction=0.55, conductivity=1.50, density=2300.0, specific_heat=880.0, emissivity=0.93),
    Material("oak", friction=0.55, conductivity=0.16, density=700.0, specific_heat=1700.0, emissivity=0.90),
    Material("abs", friction=0.40, conductivity=0.17, density=1040.0, specific_heat=1400.0, emissivity=0.94),
    Material("silicone", friction=1.20, conductivity=0.20, density=1200.0, specific_heat=1300.0, emissivity=0.90),
    Material("foam", friction=0.60, conductivity=0.035, density=30.0, specific_heat=1300.0, emissivity=0.90),
)

# The fingertip's own skin: a silicone pad over the sensing element. Its low
# conductivity is what makes every metal look alike -- the skin, not the plate,
# is the bottleneck in the series thermal resistance.
FINGERTIP_CONDUCTIVITY = 0.25
FINGERTIP_DENSITY = 1200.0
FINGERTIP_SPECIFIC_HEAT = 1300.0
FINGERTIP_EMISSIVITY = 0.05

# The tactile board runs warm; plates and air sit at room temperature.
FINGERTIP_TEMPERATURE = 35.0
AMBIENT_TEMPERATURE = 22.0
