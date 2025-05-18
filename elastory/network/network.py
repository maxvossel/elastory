from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
import yaml

from elastory.response.qopt import concentric, guided
from elastory.utils.hessian import calc_hessian
from elastory.utils.matrix import connected_beads, connectivity, laplacian
from elastory.utils.repulsive import dense_square_map


class Variant(Enum):
    SYMMETRIC = "symmetric"
    ANTISYMMETRIC = "antisymmetric"


class Kind(Enum):
    PROTEIN = "protein"
    PSEUDOPROTEIN = "trained network"
    LITERATURE = "literature"


class Optimizer(Enum):
    CONCENTRIC = "concentric"
    GUIDED = "guided"


@dataclass
class NetworkGeometry:
    """
    Class to store the geometry of a network.
    """

    bead_positions: npt.NDArray[np.float64]
    cutoff_length: float
    epsilon: float | None
    repulsive_cutoff: float | None

    def __post_init__(self):
        self.number_of_beads = self.bead_positions.shape[0]
        self.dimension_spatial = self.bead_positions.shape[1]
        self.dimension_full = self.number_of_beads * self.dimension_spatial

        self.laplacian = laplacian(self.bead_positions, self.cutoff_length)
        self.connectivity = connectivity(self.laplacian)
        self.connected = connected_beads(self.connectivity)
        self.eq_dist = self.calculate_eq_dist()
        self.hessian, self.dist_mat, self.diff_mat = calc_hessian(
            self.bead_positions, self.laplacian
        )
        self.dist_mat_connected = self.dist_mat * self.connectivity

        self.sigma = self.repulsive_cutoff / 2 ** (1 / 6) if self.repulsive_cutoff else None

    def calculate_eq_dist(self) -> npt.NDArray[np.float64]:
        vec = self.bead_positions[self.connected[0]] - self.bead_positions[self.connected[1]]
        return np.sqrt(np.sum(np.power(vec, 2), axis=1))


@dataclass
class NetworkGraph:
    """
    Class to build a network graph from a network geometry.
    """

    geometry: NetworkGeometry
    graph: nx.Graph = field(init=False)

    def __post_init__(self):
        self.graph = self.build_graph()

    def build_graph(self) -> nx.Graph:
        G = nx.Graph()
        self._add_nodes(G)
        self._add_edges(G)
        return G

    def _add_nodes(self, graph: nx.Graph) -> None:
        """Add beads as nodes to the graph"""
        for u, p in enumerate(self.geometry.bead_positions):
            graph.add_node(u, pos=p, color="rgba(100,100,100,0.9)", size=150)

    def _add_edges(self, graph: nx.Graph) -> None:
        """Add connections between nodes that have a smaller distance than
        the cutoff length"""
        combs = list(combinations(range(self.geometry.number_of_beads), 2))
        for o, d in combs:
            if self.geometry.dist_mat[o, d] < self.geometry.cutoff_length:
                graph.add_edge(o, d, dist=self.geometry.dist_mat[o, d])


class NetworkResponse:
    """
    Class to calculate the response of a network.
    """

    def __init__(self):
        self.response_data: dict[str, Any] = {}
        self.response_data_swapped_beads: dict[str, Any] = {}
        self.lin_response_data: dict[str, Any] = {}
        self.lin_response_data_swapped_beads: dict[str, Any] = {}

    def calculate_response(
        self, network: "Network", optimizer: str, adaptive_connections: bool, **kwargs
    ):
        # make sure that the necessary parameters are set
        assert network.max_pull is not None, "max_pull must be set"
        assert network.source is not None, "source must be set"

        default_options = {
            "optimizer": optimizer,
            "bead_positions": network.geometry.bead_positions,
            "max_pull": network.max_pull,
            "beads": network.source,
            "adaptive_connections": adaptive_connections,
            "laplacian": network.geometry.laplacian,
            "swap_beads": False,
            "repulsive_beads": False,
            "keep_eq_dist": True,
            "eq_dist": network.geometry.eq_dist,
            "connected": network.geometry.connected,
            "suppress_NullSpaceError": False,
            "mapping": network.dense_square_map,
            "epsilon": network.geometry.epsilon,
            "sigma": network.geometry.sigma,
            "repulsive_cutoff": network.geometry.repulsive_cutoff,
            "cutoff_length": network.geometry.cutoff_length,
        }

        default_options.update(kwargs)

        if default_options["swap_beads"]:
            default_options["beads"] = network.target
            default_options["variant"] = network.variant

        match optimizer:
            case "concentric":
                response_data = concentric(**default_options)
            # TODO: implement swapping of beads
            case "guided":
                pos_to_follow = kwargs.get("pos_to_follow")
                assert pos_to_follow is not None, "pos_to_follow must be provided"
                response_data = guided(pos_to_follow, **default_options)
        return response_data


class Network:
    def __init__(
        self,
        identifier: str,
        bead_positions: npt.NDArray[np.float64],
        cutoff_length: float = 9.0,
        epsilon: float | None = None,
        repulsive_cutoff: float | None = None,
        variant: Variant | str | None = None,
        max_pull: float | None = None,
        source: list[int] | None = None,
        target: list[int] | None = None,
        source_pairs: list[list[int]] | None = None,
        kind: Kind | None = None,
    ):
        """
        Class for a beads-on-springs network.
        Parameters:
        ------------
        identifier : str
            String identifier of the network.
        bead_positions : (N, D) np.ndarray
            Positions of the beads in the network.
        cutoff_length : float
        epsilon : float
        repulsive_cutoff : float
        variant : Optional[Variant | str]
            Variant kind, either 'symmetric' or 'antisymmetric'.
        max_pull : Optional[float]
            Maximum pull of the beads towards the center of mass.
        """
        self.identifier = identifier
        self.variant = self.set_variant(variant) if variant else None
        self.kind = self.set_kind(kind) if kind else None
        self._max_pull = max_pull

        self.geometry = NetworkGeometry(
            bead_positions,
            cutoff_length,
            epsilon,
            repulsive_cutoff,
        )
        self.graph = NetworkGraph(self.geometry)
        self.response = NetworkResponse()

        self._source: list[int] | None = source
        self._target: list[int] | None = target
        self._source_pairs: list[list[int]] | None = source_pairs

        self._dense_square_map = dense_square_map(self.geometry.number_of_beads)

    def set_variant(self, variant: str | Variant):
        """
        Set the variant of the network.

        Parameters:
        - variant (str): Variant kind, either 'symmetric' or 'antisymmetric'.
        """
        try:
            self.variant = Variant(variant)
        except ValueError:
            raise ValueError(
                f"Invalid variant: {variant}. Must be one of {[v.value for v in Variant]}"
            )

    def set_kind(self, kind: str | Kind):
        """
        Set the kind of the network.

        Parameters:
        - kind (str): Kind, one of 'protein' or 'trained network' or 'literature'.
        """
        try:
            self.kind = Kind(kind)
        except ValueError:
            raise ValueError(f"Invalid kind: {kind}. Must be one of {[v.value for v in Kind]}")

    @property
    def dense_square_map(self):
        """
        Mapping function for dense square matrix.
        Maps a 1D array of indices to a 2D array of indices.
        """
        return self._dense_square_map

    @property
    def N(self) -> int:
        """Number of beads of the network"""
        return self.geometry.number_of_beads

    @property
    def D(self) -> int:
        """Spatial dimension of the network"""
        return self.geometry.dimension_spatial

    @property
    def ND(self) -> int:
        """Full dimension of the network, (N * D)"""
        return self.geometry.dimension_full

    @property
    def laplacian(self) -> npt.NDArray[np.float64]:
        """Laplacian matrix of the network"""
        return self.geometry.laplacian

    @property
    def connectivity(self) -> npt.NDArray[np.float64]:
        """Connectivity matrix of the network"""
        return self.geometry.connectivity

    @property
    def connected(self):
        """Connected beads of the network"""
        return self.geometry.connected

    @property
    def hessian(self) -> npt.NDArray[np.float64]:
        """Hessian matrix of the network"""
        return self.geometry.hessian

    @property
    def max_pull(self) -> float | None:
        return self._max_pull

    @max_pull.setter
    def max_pull(self, value: float):
        if 0 <= value <= 1:
            self._max_pull = value
        else:
            raise ValueError("max_pull must be between 0 and 1")

    @property
    def source(self) -> list[int] | None:
        return self._source

    @source.setter
    def source(self, value: list[int]):
        assert all([v in range(self.N) for v in value]), "Bead index out of range for target"
        self._source = value
        for n in self._source:
            self.graph.graph.nodes[n]["color"] = "darkblue"
            self.graph.graph.nodes[n]["size"] = 300
        if len(self._source) == 2:
            self._source_pairs = [self._source]

    @property
    def target(self) -> list[int] | None:
        return self._target

    @target.setter
    def target(self, value: list[int]):
        assert all([v in range(self.N) for v in value]), "Bead index out of range for target"
        self._target = value
        for n in self._target:
            self.graph.graph.nodes[n]["color"] = "darkred"
            self.graph.graph.nodes[n]["size"] = 300

    @property
    def source_pairs(self) -> list[list[int]] | None:
        return self._source_pairs

    @source_pairs.setter
    def source_pairs(self, value: list[list[int]]):
        self._source_pairs = value

    ### Serialization methods
    def to_dict(self):
        """
        Convert the Network object to a dictionary for serialization.
        """
        return {
            "identifier": self.identifier,
            "bead_positions": self.geometry.bead_positions.tolist(),
            "cutoff_length": self.geometry.cutoff_length,
            "epsilon": self.geometry.epsilon,
            "repulsive_cutoff": self.geometry.repulsive_cutoff,
            "variant": self.variant.value if self.variant else None,
            "max_pull": self._max_pull,
            "source": self._source,
            "target": self._target,
            "source_pairs": np.array(self._source_pairs).tolist() if self._source_pairs else None,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create a Network object from a dictionary.
        """
        return cls(
            identifier=data["identifier"],
            bead_positions=np.array(data["bead_positions"]),
            cutoff_length=data["cutoff_length"],
            epsilon=data["epsilon"],
            repulsive_cutoff=data["repulsive_cutoff"],
            variant=data["variant"],
            max_pull=data["max_pull"],
            source=data["source"],
            target=data["target"],
            source_pairs=data["source_pairs"],
        )

    def save_to_file(self, filename):
        """
        Save the Network object to a YAML file.
        """
        with open(filename, "w") as file:
            yaml.dump(self.to_dict(), file, default_flow_style=False)

    @classmethod
    def load_from_file(cls, filename):
        """
        Load a Network object from a YAML file.
        """
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
        net = cls.from_dict(data)
        net.source = data["source"]
        net.target = data["target"]
        return net

    ### convenience wrappers for response calculation methods
    def calculate_response_concentric(
        self,
        adaptive_connections: bool = False,
        **kwargs: Any,
    ):
        """
        Calculate the response using the simplified 'QOPT' method,
        pulling the beads towards their shared center of mass/gravity.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments for the QOPT calculation.

        Returns
        -------
        trajectory[npt.NDArray[np.float64]]
            Trajectory of the bead positions
        """
        return self.response.calculate_response(
            self,
            Optimizer.CONCENTRIC.value,
            adaptive_connections,
            **kwargs,
        )

    def calculate_response_guided(
        self,
        pos_to_follow: npt.NDArray[np.float64],
        adaptive_connections: bool = False,
        **kwargs: Any,
    ):
        """
        Calculate the response using the simplified 'QOPT' method,
        pulling the beads along predefined preload positions (trajectories).

        Parameters
        ----------
        pos_ind_preload : npt.NDArray[np.float64]
            Predefined preload positions for the beads.
        **kwargs : Any
            Additional keyword arguments for the QOPT calculation.

        Returns
        -------
        trajectory[npt.NDArray[np.float64]]
            Trajectory of the bead positions
        """
        return self.response.calculate_response(
            self,
            Optimizer.GUIDED.value,
            adaptive_connections,
            pos_to_follow=pos_to_follow,
            **kwargs,
        )
