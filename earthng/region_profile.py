"""Region profiles of the exporter: EARTHNG_PROFILE=fork switches on what the fork adds on top of the
upstream exporter, as one setting.

Without a profile nothing here acts and a pack comes out as the upstream exporter makes it. With
EARTHNG_PROFILE=fork
  * the exporter switches that default off (EARTHNG_ROAD_RULES, EARTHNG_POWER, EARTHNG_NOISE,
    EARTHNG_CAMERAS) default to on; a variable set in the environment still wins, and
  * every region manifest carries `extras`, the Lua generator's switches for this pack
    (generator.lua ROADS.extras; native.lua applies them when the landing starts): signals, lots,
    roadProfiles and lampRows on, and openSea on (the sea plane then floods only the sea joined to a
    seed, so marsh and polder land below the sea level stays dry).
A pack without `extras` leaves the generator at its defaults.
"""
import os

FORK_ENV = {'EARTHNG_ROAD_RULES': '1', 'EARTHNG_POWER': '1', 'EARTHNG_NOISE': '1', 'EARTHNG_CAMERAS': '1'}
FORK_EXTRAS = {'signals': True, 'lots': True, 'roadProfiles': True, 'lampRows': True, 'openSea': True}


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
