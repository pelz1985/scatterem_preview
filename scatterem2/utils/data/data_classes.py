from dataclasses import dataclass
from typing import Optional, Union

import h5py
import numpy as np
import torch
import yaml
from numpy.fft import fftshift
from numpy.typing import NDArray
from torch import Tensor

from scatterem2.io.serialize import AutoSerialize
from scatterem2.utils import (
    beamlet_samples,
    energy2wavelength,
    fftfreq2,
    natural_neighbor_weights,
)
from scatterem2.utils.data.abberations import Aberrations
from scatterem2.utils.validators import validate_ndinfo, validate_units


@dataclass
class Geometry:
    translation: Tensor
    rotation: Tensor

    def __str__(self) -> str:
        """Pretty print all class members"""
        return (
            "Geometry:\n"
            f"  translation:    {self.translation.cpu().numpy()}\n"
            f"  rotation:    {self.rotation.cpu().numpy()}"
        )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class MetadataNew(AutoSerialize):
    """Base metadata class containing physical parameters for electron microscopy.

    Attributes:
        semiconvergence_angle (float): Beam convergence semi-angle in radians.
        energy (float): Electron beam energy in electron volts.
        rotation (float): Sample rotation angle in degrees.
        defocus_guess (float): Initial guess of sample thickness in Angstroms.
        sample_thickness_guess (float): Initial guess of sample thickness in Angstroms.
    """

    energy: float

    defocus_guess: float
    sample_thickness_guess: float

    @property
    def wavelength(self) -> float:
        """Calculate electron wavelength from energy.

        Returns:
            float: Electron wavelength in Angstroms.
        """
        return energy2wavelength(self.energy)

    def __str__(self) -> str:
        """Pretty print all class members"""
        return (
            "Metadata:\n"
            f"  energy: {self.energy}\n"
            f"  defocus_guess: {self.defocus_guess}\n"
            f"  sample_thickness_guess: {self.sample_thickness_guess}"
        )


@dataclass
class Metadata:
    """Base metadata class containing physical parameters for electron microscopy.

    Attributes:
        dk (np.ndarray): Reciprocal space sampling in inverse Angstroms.
        semiconvergence_angle (float): Beam convergence semi-angle in radians.
        energy (float): Electron beam energy in electron volts.
        rotation (float): Sample rotation angle in degrees.
        aberrations (Aberrations): Lens aberration coefficients.
        sample_thickness_guess (float): Initial guess of sample thickness in Angstroms.
    """

    dk: np.ndarray
    semiconvergence_angle: float
    energy: float
    rotation: float
    aberrations: Aberrations
    sample_thickness_guess: float

    @property
    def wavelength(self) -> float:
        """Calculate electron wavelength from energy.

        Returns:
            float: Electron wavelength in Angstroms.
        """
        return energy2wavelength(self.energy)

    @property
    def dalpha(self) -> float:
        """Calculate angular sampling from reciprocal space sampling and wavelength.

        Returns:
            float: Angular sampling in radians.
        """
        return self.dk * self.wavelength


@dataclass
class Metadata4D(Metadata):
    """Metadata class containing physical parameters and scan geometry for 4D-STEM.

    Extends the base Metadata class with additional attributes specific to 4D-STEM scanning.

    Attributes:
        scan_step (Tensor): Real space scan step size in Angstroms.
        num_scan_steps (Tensor): Number of scan positions in each direction.
        scan_step_pixels (Tensor): Real space scan step size in detector pixels.
        detector_shape (Tensor): Shape of detector in pixels.
        vacuum_probe (Tensor): Complex vacuum probe wavefunction.
    """

    scan_step: Tensor
    num_scan_steps: Tensor
    scan_step_pixels: Tensor
    detector_shape: Tensor
    vacuum_probe: Tensor

    def __init__(
        self,
        energy: float,
        semiconvergence_angle: float,
        rotation: Optional[float] = 0,
        dk: Optional[Tensor] = Tensor([0.0, 0.0]),
        aberrations: Optional[Aberrations] = Aberrations(torch.zeros((12,))),
        sample_thickness_guess: Optional[float] = 0,
        vacuum_probe: Optional[Tensor] = None,
        scan_step: Optional[Tensor] = Tensor([0.0, 0.0]),
        num_scan_steps: Optional[Tensor] = Tensor([0.0, 0.0]),
        scan_step_pixels: Optional[Tensor] = Tensor([0.0, 0.0]),
        detector_shape: Optional[Tensor] = Tensor([0, 0]),
    ) -> None:
        """Initialize Metadata4D object with experimental parameters.

        Args:
            energy (float): Electron beam energy in electron volts.
            semiconvergence_angle (float): Beam convergence semi-angle in radians.
            rotation (Optional[float], optional): Sample rotation angle in degrees. Defaults to 0.
            dk (Optional[Tensor], optional): Reciprocal space sampling in inverse Angstroms. Defaults to [0.0, 0.0].
            aberrations (Optional[Aberrations], optional): Lens aberration coefficients. Defaults to zeros.
            sample_thickness_guess (Optional[float], optional): Initial guess of sample thickness in Angstroms. Defaults to 0.
            vacuum_probe (Optional[Tensor], optional): Complex vacuum probe wavefunction. Defaults to None.
            scan_step (Optional[Tensor], optional): Real space scan step size in Angstroms. Defaults to [0.0, 0.0].
            num_scan_steps (Optional[Tensor], optional): Number of scan positions in each direction. Defaults to [0.0, 0.0].
            scan_step_pixels (Optional[Tensor], optional): Real space scan step size in detector pixels. Defaults to [0.0, 0.0].
            detector_shape (Optional[Tensor], optional): Shape of detector in pixels. Defaults to [0, 0].
        """

        self.scan_step = scan_step
        self.dk = dk
        self.semiconvergence_angle = semiconvergence_angle
        self.rotation = rotation
        self.energy = energy
        self.aberrations = aberrations
        self.scan_step_pixels = scan_step_pixels
        self.sample_thickness_guess = sample_thickness_guess
        self.num_scan_steps = num_scan_steps
        self.detector_shape = detector_shape
        self.vacuum_probe = vacuum_probe

    @property
    def k_max(self) -> float:
        """Calculate maximum scattering vector magnitude from semiconvergence angle and detector shape.

        Returns:
            float: Maximum scattering vector magnitude in inverse Angstroms.
        """
        return self.dk * self.detector_shape / 2

    @property
    def dr(self) -> float:
        """Calculate real space sampling from semiconvergence angle and detector shape.

        Returns:
            float: Real space sampling in Angstroms.
        """
        return 1 / (2 * self.k_max)

    def to_h5(self, file_path: str, key: str = "meta") -> None:
        """Save metadata to HDF5 file.

        Args:
            file_path (str): Path to HDF5 file.
            key (str, optional): Group name in HDF5 file. Defaults to "meta".
        """
        with h5py.File(file_path, "a") as f:
            g = f.create_group(key)
            g.create_dataset("scan_step", data=self.scan_step)
            g.create_dataset("dk", data=self.dk)
            g.create_dataset("alpha_rad", data=self.semiconvergence_angle)
            g.create_dataset("rotation_deg", data=self.rotation)
            g.create_dataset("E_ev", data=self.energy)
            g.create_dataset("wavelength", data=self.wavelength)
            g.create_dataset("aberrations", data=self.aberrations)
            g.create_dataset("pixel_step", data=self.scan_step_pixels)
            g.create_dataset(
                "sample_thickness_guess_angstrom",
                data=self.sample_thickness_guess,
            )

    def to_dict(self) -> dict:
        """Convert metadata to dictionary.

        Returns:
            dict: Dictionary containing metadata values.
        """
        ret = {}
        ret["scan_step"] = list(self.scan_step.astype(np.float64))
        ret["dk"] = list(self.dk.astype(np.float64))
        ret["alpha_rad"] = self.semiconvergence_angle
        ret["rotation_deg"] = self.rotation
        ret["E_ev"] = self.energy
        ret["wavelength"] = self.wavelength
        ret["aberrations"] = list(self.aberrations.astype(np.float64))
        ret["pixel_step"] = list(self.scan_step_pixels.astype(np.float64))
        ret["sample_thickness_guess_angstrom"] = self.sample_thickness_guess
        return ret

    @classmethod
    def from_dict(cls, dictionary: dict) -> "Metadata4D":
        """Create Metadata4D instance from dictionary.

        Args:
            dictionary (dict): Dictionary containing metadata values.

        Returns:
            Metadata4D: New instance initialized from dictionary.
        """
        res = cls()
        res.scan_step = np.array(dictionary["scan_step"])
        res.dk = np.array(dictionary["dk"])
        res.semiconvergence_angle = dictionary["alpha_rad"]
        res.rotation = dictionary["rotation_deg"]
        res.energy = dictionary["E_ev"]
        res.wavelength = dictionary["wavelength"]
        res.aberrations = Aberrations(torch.as_tensor(dictionary["aberrations"]))
        res.scan_step_pixels = torch.as_tensor(dictionary["pixel_step"])
        res.sample_thickness_guess = dictionary["sample_thickness_guess_angstrom"]
        return res

    @classmethod
    def from_h5(cls, file_path: str, key: str = "meta") -> "Metadata4D":
        """Create Metadata4D instance from HDF5 file.

        Args:
            file_path (str): Path to HDF5 file.
            key (str, optional): Group name in HDF5 file. Defaults to "meta".

        Returns:
            Metadata4D: New instance initialized from HDF5 file.
        """
        res = cls()
        with h5py.File(file_path, "r") as f:
            g = f[key]
            res.scan_step = g["scan_step"][...]
            res.scan_step_pixels = g["pixel_step"][...]
            res.dk = g["dk"][...]
            res.semiconvergence_angle = g["alpha_rad"][()]
            res.rotation = g["rotation_deg"][()]
            res.energy = g["E_ev"][()]
            res.wavelength = g["wavelength"][()]
            res.aberrations = g["aberrations"][...]
            try:
                res.sample_thickness_guess = g["sample_thickness_guess_angstrom"][()]
            except KeyError:
                pass
        return res

    def __str__(self) -> str:
        """Generate string representation of metadata.

        Returns:
            str: Formatted string containing metadata values.
        """
        return (
            "Metadata4D:\n"
            f"  scan_step:     {self.scan_step.cpu().numpy()}\n"
            f"  dk:            {self.dk.cpu().numpy()}\n"
            f"  alpha_rad:     {self.semiconvergence_angle}\n"
            f"  rotation_deg:  {self.rotation:2.2f}\n"
            f"  E_ev:          {self.energy:2.2f}\n"
            f"  wavelength:    {self.wavelength:2.2f}\n"
            f"  aberrations:   {self.aberrations.array.cpu().numpy()}\n"
            f"  pixel_step:    {self.scan_step_pixels.cpu().numpy()}\n"
            f"  thicknessguess:{self.sample_thickness_guess:2.2f}"
        )

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: Same as __str__().
        """
        return self.__str__()


@dataclass
class Metadata4dstem(MetadataNew):
    """Metadata class containing physical parameters and scan geometry for 4D-STEM.

    Extends the base Metadata class with additional attributes specific to 4D-STEM scanning.

    Attributes:
        vacuum_probe (Tensor): Complex vacuum probe wavefunction.
    """

    vacuum_probe: Optional[Tensor]
    semiconvergence_angle: float
    rotation: float
    aberrations: Optional[Aberrations]

    def __init__(
        self,
        energy: float,
        semiconvergence_angle: float,
        sampling: Union[NDArray, tuple, list, float, int],
        units: Union[list[str], tuple, list],
        shape: NDArray,
        rotation: Optional[float] = 0,
        defocus_guess: Optional[float] = 0,
        aberrations: Optional[Aberrations] = Aberrations(torch.zeros((12,))),
        slice_thickness: Optional[float] = 0,
        sample_thickness_guess: Optional[float] = 0,
        vacuum_probe: Optional[Tensor] = None,
    ) -> None:
        """Initialize Metadata4D object with experimental parameters.

        Args:
            energy (float): Electron beam energy in electron volts.
            semiconvergence_angle (float): Beam convergence semi-angle in radians.
            rotation (Optional[float], optional): Sample rotation angle in degrees. Defaults to 0.
            defocus_guess (Optional[float], optional): Initial guess of sample thickness in Angstroms. Defaults to 0.
            sample_thickness_guess (Optional[float], optional): Initial guess of sample thickness in Angstroms. Defaults to 0.
            vacuum_probe (Optional[Tensor], optional): Complex vacuum probe wavefunction. Defaults to None.


        """
        super().__init__(energy, defocus_guess or 0, sample_thickness_guess or 0)
        self.vacuum_probe = vacuum_probe
        self.semiconvergence_angle = semiconvergence_angle
        self.rotation = rotation or 0
        self.sampling = validate_ndinfo(
            sampling, len(sampling) if hasattr(sampling, "__len__") else 4, "sampling"
        )
        self.units = validate_units(
            units, len(sampling) if hasattr(sampling, "__len__") else 4
        )
        self.ndim = len(self.sampling)
        self.shape = shape
        self.aberrations = aberrations
        self.slice_thickness = slice_thickness

    @property
    def detector_shape(self) -> NDArray:
        """ """
        return np.array(self.shape[-2:])

    @property
    def k_max(self) -> NDArray:
        """Calculate maximum scattering vector magnitude from semiconvergence angle and detector shape.

        Returns:
            float: Maximum scattering vector magnitude in inverse Angstroms.
        """
        return self.sampling[-2:] * self.detector_shape / 2

    @property
    def dr(self) -> NDArray:
        """Calculate real space sampling of the detector from k_max.

        Returns:
            float: Real space sampling of the detector in Angstroms.
        """
        return 1 / (2 * self.k_max)

    def to_yaml(self, file_path: str) -> None:
        """Save metadata to YAML file.

        This method updates the YAML file by preserving existing entries and only
        updating the ones that exist in the metadata dictionary.

        Args:
            file_path (str): Path to YAML file to update.
        """
        from pathlib import Path

        # Convert metadata to dictionary
        metadata_dict = self.to_dict()
        dict = {"meta4d": metadata_dict}

        # Load existing YAML content if file exists
        existing_data = {}
        if Path(file_path).exists():
            try:
                with open(file_path, "r") as f:
                    existing_data = yaml.safe_load(f) or {}
            except (yaml.YAMLError, FileNotFoundError):
                # If file is corrupted or doesn't exist, start with empty dict
                existing_data = {}
        print(f"existing_data: {existing_data}")
        # Update existing data with new metadata
        # This preserves all existing entries and only updates/overwrites
        # the ones that exist in metadata_dict
        existing_data.update(dict)

        # Write back the updated data
        with open(file_path, "w") as f:
            yaml.dump(existing_data, f, default_flow_style=False)

    def to_h5(self, file_path: str, key: str = "meta") -> None:
        """Save metadata to HDF5 file.

        Args:
            file_path (str): Path to HDF5 file.
            key (str, optional): Group name in HDF5 file. Defaults to "meta".
        """
        with h5py.File(file_path, "a") as f:
            g = f.create_group(key)
            g.create_dataset("alpha_rad", data=self.semiconvergence_angle)
            g.create_dataset("rotation_deg", data=self.rotation)
            g.create_dataset("E_ev", data=self.energy)
            g.create_dataset("wavelength", data=self.wavelength)
            g.create_dataset("defocus_guess_angstrom", data=self.defocus_guess)
            g.create_dataset(
                "sample_thickness_guess_angstrom",
                data=self.sample_thickness_guess,
            )
            if self.vacuum_probe is not None:
                g.create_dataset("vacuum_probe", data=self.vacuum_probe)

    def to_dict(self) -> dict:
        """Convert metadata to dictionary.

        Returns:
            dict: Dictionary containing metadata values.
        """
        ret = {}
        ret["alpha_rad"] = float(self.semiconvergence_angle)
        ret["rotation_deg"] = float(self.rotation)
        ret["E_ev"] = float(self.energy)
        ret["wavelength"] = float(self.wavelength)
        ret["defocus_guess_angstrom"] = float(self.defocus_guess)
        ret["sample_thickness_guess_angstrom"] = float(self.sample_thickness_guess)
        ret["sampling"] = self.sampling.tolist()
        ret["units"] = self.units
        ret["shape"] = self.shape.tolist()
        if self.aberrations is not None:
            ret["aberrations"] = self.aberrations.array.cpu().numpy().tolist()
        if self.vacuum_probe is not None:
            ret["vacuum_probe"] = self.vacuum_probe.cpu().numpy().tolist()
        return ret

    @classmethod
    def from_dict(cls, dictionary: dict) -> "Metadata4dstem":
        """Create Metadata4dstem instance from dictionary.

        Args:
            dictionary (dict): Dictionary containing metadata values.

        Returns:
            Metadata4dstem: New instance initialized from dictionary.
        """
        energy = dictionary["E_ev"]
        semiconvergence_angle = dictionary["alpha_rad"]
        rotation = dictionary["rotation_deg"]
        defocus_guess = dictionary.get("defocus_guess_angstrom", 0.0)
        sample_thickness_guess = dictionary.get("sample_thickness_guess_angstrom", 0.0)
        sampling = dictionary.get("sampling", np.ones(4))
        units = dictionary.get("units", ["pixels"] * 4)
        shape = dictionary.get("shape", None)
        slice_thickness = dictionary.get("slice_thickness", 0.0)
        if shape is not None and isinstance(shape, list):
            shape = np.array(shape)
        aberrations = Aberrations(
            torch.as_tensor(dictionary.get("aberrations", np.zeros((12,))))
        )
        vacuum_probe = None
        if "vacuum_probe" in dictionary:
            vacuum_probe = torch.as_tensor(dictionary["vacuum_probe"])

        return cls(
            energy=energy,
            semiconvergence_angle=semiconvergence_angle,
            rotation=rotation,
            defocus_guess=defocus_guess,
            sample_thickness_guess=sample_thickness_guess,
            vacuum_probe=vacuum_probe,
            sampling=sampling,
            units=units,
            shape=shape,
            aberrations=aberrations,
            slice_thickness=slice_thickness,
        )

    @classmethod
    def from_h5(cls, file_path: str, key: str = "meta") -> "Metadata4dstem":
        """Create Metadata4dstem instance from HDF5 file.

        Args:
            file_path (str): Path to HDF5 file.
            key (str, optional): Group name in HDF5 file. Defaults to "meta".

        Returns:
            Metadata4dstem: New instance initialized from HDF5 file.
        """
        with h5py.File(file_path, "r") as f:
            g = f[key]
            energy = g["E_ev"][()]
            semiconvergence_angle = g["alpha_rad"][()]
            rotation = g["rotation_deg"][()]
            defocus_guess = g.get("defocus_guess_angstrom", 0.0)[()]
            sample_thickness_guess = g.get("sample_thickness_guess_angstrom", 0.0)[()]
            sampling = g.get("sampling", np.ones(4))
            units = g.get("units", ["pixels"] * 4)
            shape = g.get("shape", [1, 1, 1, 1])
            if isinstance(shape, list):
                shape = np.array(shape)
            aberrations = Aberrations(
                torch.as_tensor(g.get("aberrations", np.zeros((12,))))
            )
            vacuum_probe = None
            if "vacuum_probe" in g:
                vacuum_probe = torch.as_tensor(g["vacuum_probe"][...])

        return cls(
            energy=energy,
            semiconvergence_angle=semiconvergence_angle,
            rotation=rotation,
            defocus_guess=defocus_guess,
            sample_thickness_guess=sample_thickness_guess,
            vacuum_probe=vacuum_probe,
            sampling=sampling,
            units=units,
            shape=shape,
            aberrations=aberrations,
        )

    def __str__(self) -> str:
        """Generate string representation of metadata.

        Returns:
            str: Formatted string containing metadata values.
        """
        base_str = (
            "Metadata4dstem:\n"
            f"  alpha_rad:     {self.semiconvergence_angle}\n"
            f"  rotation_deg:  {self.rotation:2.2f}\n"
            f"  E_ev:          {self.energy:2.2f}\n"
            f"  wavelength:    {self.wavelength:2.2f}\n"
            f"  defocus_guess: {self.defocus_guess:2.2f}\n"
            f"  sample_thickness_guess: {self.sample_thickness_guess:2.2f}\n"
            f"  sampling:      {self.sampling}\n"
            f"  units:         {self.units}\n"
            f"  shape:         {self.shape}\n"
        )
        if self.aberrations is not None:
            base_str += f"\n  aberrations:   {self.aberrations.array.cpu().numpy()}\n"
        if self.vacuum_probe is not None:
            base_str += f"\n  vacuum_probe:  {self.vacuum_probe.shape}"
        return base_str

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: Same as __str__().
        """
        return self.__str__()


@dataclass
class Metadata4dstemAndGeometry(Metadata4dstem):
    geometry: Geometry

    def __init__(
        self,
        energy: float,
        semiconvergence_angle: float,
        rotation: Optional[float] = 0,
        defocus_guess: Optional[float] = 0,
        sample_thickness_guess: Optional[float] = 0,
        vacuum_probe: Optional[Tensor] = None,
        geometry: Optional[Geometry] = None,
    ) -> None:
        super().__init__(
            energy,
            semiconvergence_angle,
            rotation,
            defocus_guess,
            sample_thickness_guess,
            vacuum_probe,
        )
        self.geometry = geometry

    def __str__(self) -> str:
        """Pretty print all class members"""
        base_str = super().__str__()
        if self.geometry is not None:
            base_str += f"\n{self.geometry}"
        return base_str

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Metadata4DAndGeometry:
    meta4d: Metadata4D
    geometry: Geometry

    def __init__(self, meta4d: Metadata4D, geometry: Geometry) -> None:
        self.meta4d = meta4d
        self.geometry = geometry

    def __str__(self) -> str:
        """Pretty print all class members"""
        return f"Metadata4DAndGeometry:\n{self.meta4d}\n{self.geometry}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class SMeta:
    f: np.ndarray
    M: np.ndarray
    N: np.ndarray
    dx: np.ndarray

    q: Tensor
    qf: Tensor
    q2: Tensor
    qf2: Tensor
    q_coords: Tensor
    r_indices: Tensor
    all_beams: Tensor
    parent_beams: Tensor
    beam_numbers: Tensor
    all_beams_q: Tensor
    all_beams_coords: Tensor
    parent_beams_q: Tensor
    parent_beams_coords: Tensor
    q_dft: Tensor

    numerical_aperture_radius_pixels: int
    B: int
    Bp: int

    device: torch.device

    def __init__(
        self,
        all_beams: np.ndarray,
        parent_beams: np.ndarray,
        numerical_aperture_radius_pixels: int,
        dx: np.ndarray,
        M: tuple[int, int],
        device: torch.device,
    ) -> None:
        self.dx = dx
        self.device = device
        self.all_beams = torch.as_tensor(all_beams).to(device).bool()
        self.parent_beams = torch.as_tensor(parent_beams).to(device).bool()
        self.numerical_aperture_radius_pixels = numerical_aperture_radius_pixels
        self.B = int(all_beams.sum())
        self.Bp = int(parent_beams.sum())

        MY, MX = M
        self.M = np.array(M)

        self.qf = fftfreq2([MY, MX], dx, centered=False, device=device)

        self.q2 = torch.norm(self.q, dim=0) ** 2
        self.qf2 = torch.norm(self.qf, dim=0) ** 2
        mgrid = np.array(np.mgrid[-MY // 2 : MY // 2, -MX // 2 : MX // 2])
        self.q_coords = torch.from_numpy(fftshift(mgrid, (1, 2))).to(device)
        self.r_indices = torch.from_numpy(np.mgrid[:MY, :MX]).to(device)

        self.beam_numbers = (
            torch.ones_like(self.all_beams, dtype=torch.long, device=device) * -1
        )
        self.beam_numbers[self.all_beams] = torch.arange(0, self.B, device=device)

        all_beams_expanded = self.all_beams[None, ...].expand_as(self.q_coords)
        self.all_beams_q = self.qf[all_beams_expanded].reshape(2, self.B).T
        self.all_beams_coords = self.q_coords[all_beams_expanded].reshape(2, self.B).T

        parent_beams_expanded = self.parent_beams[None, ...].expand_as(self.q_coords)
        self.parent_beams_q = self.qf[parent_beams_expanded].reshape(2, self.Bp).T
        self.parent_beams_coords = (
            self.q_coords[parent_beams_expanded].reshape(2, self.Bp).T
        )

    def to_h5(self, file_path: str, key: str = "s_matrix_meta") -> None:
        with h5py.File(file_path, "a") as f:
            g = f.create_group(key)
            g.create_dataset("M", data=self.M)
            g.create_dataset("N", data=self.N)
            g.create_dataset("all_beams", data=self.all_beams.cpu().numpy())
            g.create_dataset("parent_beams", data=self.parent_beams.cpu().numpy())
            g.create_dataset(
                "numerical_aperture_radius_pixels",
                data=self.numerical_aperture_radius_pixels,
            )
            g.create_dataset("dx", data=self.dx)
            g.create_dataset("device", data=str(self.device))

    @classmethod
    def from_h5(cls, file_path: str, key: str = "s_matrix_meta") -> "SMeta":
        with h5py.File(file_path, "r") as f:
            g = f[key]
            all_beams = g["all_beams"][...]
            parent_beams = g["parent_beams"][...]
            numerical_aperture_radius_pixels = g["numerical_aperture_radius_pixels"][()]
            dx = g["dx"][...]
            N = g["N"][...]
            M = g["M"][...]
            device = torch.device(str(g["device"][...]))
        res = cls(
            all_beams, parent_beams, numerical_aperture_radius_pixels, dx, N, M, device
        )
        return res

    def make_beamlet_meta(
        self, n_radial_samples: int, n_angular_samples: int = 6
    ) -> "SMeta":
        parent_beams_coords = beamlet_samples(
            self.all_beams.cpu().numpy(),
            self.numerical_aperture_radius_pixels,
            n_angular_samples,
            n_radial_samples,
        )

        parent_beams = torch.zeros(tuple(self.M), dtype=torch.bool)
        for si in parent_beams_coords:
            parent_beams[si[0], si[1]] = 1

        Bp = parent_beams_coords.shape[0]
        mgrid = np.array(
            np.mgrid[-self.M[0] // 2 : self.M[0] // 2, -self.M[1] // 2 : self.M[1] // 2]
        )
        q_coords = torch.from_numpy(fftshift(mgrid, (1, 2)))
        parent_beams_expanded = parent_beams[None, ...].expand_as(q_coords)
        parent_beams_coords = q_coords[parent_beams_expanded].reshape(2, Bp).T

        if n_radial_samples > 1:
            nnw = natural_neighbor_weights(
                parent_beams_coords,
                self.all_beams_coords.cpu().numpy(),
                minimum_weight_cutoff=1e-2,
            )
        else:
            nnw = np.ones((self.B, 1))

        beamlets = []
        for j in range(nnw.shape[1]):
            wsample = nnw[:, j]
            ww = np.zeros(self.all_beams.shape, dtype=np.float32)
            ww[
                self.all_beams_coords[:, 1].cpu().numpy(),
                self.all_beams_coords[:, 0].cpu().numpy(),
            ] = wsample
            beamlets.append(ww)
        beamlets = np.array(beamlets)

        new_meta = SMeta(
            self.all_beams,
            parent_beams,
            self.numerical_aperture_radius_pixels,
            self.dx,
            tuple(self.N),
            tuple(self.M),
            self.device,
        )
        new_meta.natural_neighbor_weights = torch.as_tensor(nnw)
        new_meta.beamlets = torch.as_tensor(beamlets).to(new_meta.device)
        return new_meta
