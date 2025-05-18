from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from elastory.network.network import Optimizer, Variant


@dataclass
class SimulationConfig:
    # Bead selection
    source_beads: List[int]

    # Core simulation parameters
    max_pull: float = 0.5
    preload_steps: int = 50
    optimizer: Optimizer = Optimizer.CONCENTRIC
    variant: Variant = Variant.SYMMETRIC

    # Optimization options
    adaptive_connections: bool = False
    swap_beads: bool = False
    keep_eq_dist: bool = True
    repulsive_beads: bool = True
    suppress_NullSpaceError: bool = False

    # Additional options
    pos_to_follow: Optional[npt.NDArray[np.float64]] = None

    # Data saving options
    save_eigen: bool = False
    save_potential: bool = False

    def __post_init__(self):
        # if not isinstance(self.optimizer, Optimizer):
        #     raise ValueError("Optimizer must be either 'concentric' or 'guided'")

        if self.optimizer == Optimizer.GUIDED and self.pos_to_follow is None:
            raise ValueError("pos_to_follow must be provided when using guided optimizer")

        if self.optimizer == Optimizer.CONCENTRIC and not isinstance(self.variant, Variant):
            raise ValueError(
                "Variant must be either 'symmetric' or 'antisymmetric' when using concentric optimizer"
            )
