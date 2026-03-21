from pathlib import Path

from ares.consts import ALL_STRUCTURES, CHANGELING_TYPES, WORKER_TYPES
from sc2.constants import abilityid_to_unittypeid
from sc2.ids.unit_typeid import UnitTypeId

ALL_UNITS = ALL_STRUCTURES | set(abilityid_to_unittypeid.values())
EXCLUDE_FROM_COMBAT = WORKER_TYPES | CHANGELING_TYPES | {UnitTypeId.LARVA, UnitTypeId.EGG}
MAX_MICRO_ACTIONS = 80

DATA_DIR = Path("./data")
PARAMS_FILE = DATA_DIR / "params.json"
VERSION_FILE = Path("version.txt")
