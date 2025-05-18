from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd


class ResponseData:
    def __init__(self):
        self.positions = pd.DataFrame()
        self.potentials = pd.DataFrame()
        self.eigenvalues = pd.DataFrame()
        self.eigenvectors: List[Tuple[int, np.ndarray]] = []

    def add_step(self, step, positions, potential=None, eigenvalues=None, eigenvector=None):
        pos_df = pd.DataFrame(positions, columns=["x", "y", "z"])
        pos_df["step"] = step
        pos_df["bead"] = pos_df.index
        self.positions = pd.concat([self.positions, pos_df])

        if potential is not None:
            pot_df = pd.DataFrame({"step": [step], "potential": [potential]})
            self.potentials = pd.concat([self.potentials, pot_df])

        if eigenvalues is not None:
            eig_df = pd.DataFrame(eigenvalues, columns=["eigenvalue"])
            eig_df["step"] = step
            eig_df["mode"] = eig_df.index
            self.eigenvalues = pd.concat([self.eigenvalues, eig_df])

        if eigenvector is not None:
            self.eigenvectors.append((step, eigenvector))

    def save_to_hdf5(self, filename: str) -> None:
        with h5py.File(filename, "w") as f:
            f.create_dataset("positions", data=self.positions.to_numpy())
            f.create_dataset(
                "positions_columns",
                data=np.array(self.positions.columns.tolist(), dtype="S"),
            )
            f.create_dataset("potentials", data=self.potentials.to_numpy())
            f.create_dataset(
                "potentials_columns",
                data=np.array(self.potentials.columns.tolist(), dtype="S"),
            )
            f.create_dataset("eigenvalues", data=self.eigenvalues.to_numpy())
            f.create_dataset(
                "eigenvalues_columns",
                data=np.array(self.eigenvalues.columns.tolist(), dtype="S"),
            )
            eigenvectors_group = f.create_group("eigenvectors")
            for step, eigenvector in self.eigenvectors:
                eigenvectors_group.create_dataset(f"step_{step}", data=eigenvector)

    @classmethod
    def load_from_hdf5(cls, filename: str) -> "ResponseData":
        data = cls()
        with h5py.File(filename, "r") as f:
            data.positions = pd.DataFrame(
                cls._safe_get_dataset(f, "positions"),
                columns=[
                    col.decode("utf-8") for col in cls._safe_get_dataset(f, "positions_columns")
                ],
            )
            data.potentials = pd.DataFrame(
                cls._safe_get_dataset(f, "potentials"),
                columns=[
                    col.decode("utf-8") for col in cls._safe_get_dataset(f, "potentials_columns")
                ],
            )
            data.eigenvalues = pd.DataFrame(
                cls._safe_get_dataset(f, "eigenvalues"),
                columns=[
                    col.decode("utf-8") for col in cls._safe_get_dataset(f, "eigenvalues_columns")
                ],
            )
            eigenvectors_group = f.get("eigenvectors")
            if isinstance(eigenvectors_group, h5py.Group):
                data.eigenvectors = [
                    (
                        int(name.split("_")[1]),
                        cls._safe_get_dataset(eigenvectors_group, name),
                    )
                    for name in eigenvectors_group.keys()
                ]
        return data

    @staticmethod
    def _safe_get_dataset(group: h5py.Group, name: str) -> np.ndarray:
        dataset = group.get(name)
        if dataset is None:
            raise ValueError(f"Dataset '{name}' not found in HDF5 file")
        if isinstance(dataset, h5py.Dataset):
            return np.array(dataset)
        raise ValueError(f"'{name}' is not a Dataset")
