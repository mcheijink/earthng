"""Region profiles of the exporter: EARTHNG_PROFILE=fork switches on what the fork adds on top of the
upstream exporter, as one setting.

Without a profile nothing here acts and a pack comes out as the upstream exporter makes it. With
EARTHNG_PROFILE=fork
  * the exporter switches that default off (EARTHNG_ROAD_RULES, EARTHNG_POWER, EARTHNG_NOISE,
    EARTHNG_CAMERAS, EARTHNG_BRIDGE_FLOORS, EARTHNG_PCG) default to on and the two that default on
    (EARTHNG_SPANS, EARTHNG_TIDAL) default to off; a variable set in the environment still wins, and
  * every region manifest carries `extras`, the Lua generator's switches for this pack
    (generator.lua ROADS.extras; native.lua applies them when the landing starts): signals, lots,
    roadProfiles and lampRows on, and openSea on (the sea plane then floods only the sea joined to a
    seed, so marsh and polder land below the sea level stays dry).
A pack without `extras` leaves the generator at its defaults.
"""
import os

# EARTHNG_BRIDGE_FLOORS  decks over navigable water get a lower bound (fast_roads.bridge_floors)
# EARTHNG_PCG            the road profile's conjugate gradients run preconditioned and warm started (fast_profile)
# EARTHNG_SPANS          (default on) a bridge mapped in pieces keeps its deck (fast_profile.spans); the fork's result is without
# EARTHNG_TIDAL          (default on) harbour water at sea level is lowered under the sea plane (build_country._flood); the fork's result is without
FORK_ENV = {'EARTHNG_ROAD_RULES': '1', 'EARTHNG_POWER': '1', 'EARTHNG_NOISE': '1', 'EARTHNG_CAMERAS': '1',
            'EARTHNG_BRIDGE_FLOORS': '1', 'EARTHNG_PCG': '1', 'EARTHNG_SPANS': '0', 'EARTHNG_TIDAL': '0'}
FORK_EXTRAS = {'signals': True, 'lots': True, 'roadProfiles': True, 'lampRows': True, 'openSea': True}


def switch(variable, default):
    """An on/off environment switch: `default` when the variable is unset or empty."""
    value = os.environ.get(variable, '').strip().lower()
    if not value:
        return default
    return value not in ('0', 'off', 'no', 'false')


def name():
    """The profile asked for in the environment ('' for none)."""
    return os.environ.get('EARTHNG_PROFILE', '').strip().lower()


def extras():
    """The manifest's `extras` for the profile, a new dict; None when the profile sets none."""
    return dict(FORK_EXTRAS) if name() == 'fork' else None


def apply_env():
    """Default the profile's exporter switches in the environment (read when the modules load)."""
    if name() == 'fork':
        for key, value in FORK_ENV.items():
            os.environ.setdefault(key, value)
