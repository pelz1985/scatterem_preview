# %%
import os

import abtem
import ase
import h5py
import matplotlib.pyplot as plt
import numpy as np
from ase.cluster import Decahedron

abtem.config.set({"device": "cpu", "fft": "fftw"})
cluster = Decahedron("Pd", 9, 2, 0)
cluster.rotate("x", -30)
abtem.show_atoms(cluster, plane="xy")
substrate = ase.build.bulk("C", cubic=True)
# repeat diamond structure
substrate *= (12, 12, 10)
# displace atoms with a standard deviation of 50 % of the bond length
bondlength = 1.54  # Bond length
substrate.positions[:] += np.random.randn(len(substrate), 3) * 0.5 * bondlength
# wrap the atoms displaced outside the cell back into the cell
substrate.wrap()
translated_cluster = cluster.copy()
translated_cluster.cell = substrate.cell
translated_cluster.center()
translated_cluster.translate((0, 0, -25))
atoms = substrate + translated_cluster
atoms.center(axis=2, vacuum=2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
abtem.show_atoms(atoms, plane="xy", ax=ax1, title="Beam view")
abtem.show_atoms(atoms, plane="xz", ax=ax2, title="Side view")
plt.show()

frozen_phonons = abtem.FrozenPhonons(atoms, 64, sigmas=0.1)
potential = abtem.Potential(
    frozen_phonons,
    gpts=96,
    slice_thickness=2,
)
print(f"potential.sampling = {potential.sampling}")
print(f"potential.gpts = {potential.gpts}")
print(f"potential.shape = {potential.shape}")
# %%
potential2d = potential.build().array.sum(axis=(1)).mean(axis=(0)).compute()  # .get()

fig, ax = plt.subplots()
ax.imshow(potential2d)
plt.show()


# %%


# Get the directory of the current script or use current working directory for interactive mode
try:
    print(f"__file__ = {__file__}")
    print(f"os.getcwd() = {os.getcwd()}")

    # Check if we're in an interactive environment or if __file__ is not a real file
    if __file__ == "<stdin>" or __file__ == "/" or not os.path.isfile(__file__):
        # Running in interactive mode, use current working directory
        script_dir = os.getcwd()
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is not defined in interactive environments
    print(f"os.getcwd() = {os.getcwd()}")
    script_dir = os.getcwd()

print(f"Script directory: {script_dir}")

with h5py.File(script_dir + "/data/test_potential_2d_ssb.h5", "w") as f:
    f.create_dataset("data", data=potential2d)
    f.create_dataset("sampling", data=potential.sampling)

# %%
