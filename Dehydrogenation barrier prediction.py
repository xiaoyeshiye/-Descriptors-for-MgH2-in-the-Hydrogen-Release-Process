from ase.io import read
from ase.neighborlist import NeighborList
import numpy as np

# Open POSCAR file
atoms = read('POSCAR (7)')
atom_index_H1 = 26
atom_index_H2 = 38
# Obtain the positions of all atoms
positions = atoms.get_positions()
# Obtain lattice parameters
cell = atoms.get_cell()
# Calculate the distance between two atoms
distance = atoms.get_distance(atom_index_H1 - 1, atom_index_H2 - 1)
print("The distance between atoms is ", distance)
# Create a NeighborList object
cutoff_radius = 2.5
nl = NeighborList(cutoffs=[cutoff_radius / 2] * len(atoms), self_interaction=False, bothways=True)
# Update NeighborList
nl.update(atoms)
# Obtain Mg neighbors for H1
neighbors_H1, offsets_H1 = nl.get_neighbors(atom_index_H1 - 1)
# Obtain chemical element symbols for atoms with atomic number 1 and their neighbors
chemical_symbols = atoms.get_chemical_symbols()
atom_symbol_H1 = chemical_symbols[atom_index_H1]
neighbor_symbols_H1 = [chemical_symbols[n] for n in neighbors_H1]

# Find the surrounding Mg atomic numbers
mg_neighbors_H1 = [neighbors_H1[i] + 1 for i in range(len(neighbors_H1)) if neighbor_symbols_H1[i] == 'Mg']
# Print Results
print("The sequence number of Mg atoms within the range of 2.5 angstroms around the H1 atom is ", mg_neighbors_H1)
# Obtain Mg neighbors for H1
neighbors_H2, offsets_H2 = nl.get_neighbors(atom_index_H2 - 1)
# Obtain chemical element symbols for atoms with atomic number 1 and their neighbors
chemical_symbols = atoms.get_chemical_symbols()
atom_symbol_H2 = chemical_symbols[atom_index_H2]
neighbor_symbols_H2 = [chemical_symbols[n] for n in neighbors_H2]
# Find the surrounding Mg atomic numbers
mg_neighbors_H2 = [neighbors_H2[i] + 1 for i in range(len(neighbors_H2)) if neighbor_symbols_H2[i] == 'Mg']
# Print Results
print("The sequence number of Mg atoms within the range of 2.5 angstroms around the H2 atom is ", mg_neighbors_H2)

# Open the ICOHPLIST.lobster file (Please write the lobster file in a way similar to 'cohpGenerator from 1.4 to 2.5 type
# Mg type H' to generate the appropriate ICOHPLIST.lobster file)
with open("ICOHPLIST.lobster", "r") as file:
    lines = file.readlines()
icohp_H1_list = []
icohp_H2_list = []
# Traverse each row to match the maximum value of - ICOHP for H1 and H2
for atom_index_mg in mg_neighbors_H1:
    icohp = 0
    for line in lines:
        data = line.split()
        if len(data) == 8:
            index_h = data[1]
            index_mg = data[2]
            icohp_value = -float(data[7])
            a = 'H' + str(atom_index_H1)
            b = 'Mg' + str(atom_index_mg)
            if index_h == a and index_mg == b:
                icohp += icohp_value
    icohp_H1_list.append(icohp)
max_H1_value = max(icohp_H1_list)

for atom_index_mg in mg_neighbors_H2:
    icohp = 0
    for line in lines:
        data = line.split()
        if len(data) == 8:
            index_h = data[1]
            index_mg = data[2]
            icohp_value = -float(data[7])
            a = 'H' + str(atom_index_H2)
            b = 'Mg' + str(atom_index_mg)
            if index_h == a and index_mg == b:
                icohp += icohp_value
    icohp_H2_list.append(icohp)
max_H2_value = max(icohp_H2_list)
cohp_max_ave = (max_H1_value + max_H2_value) / 2
print("The average maximum -ICOHP is ", cohp_max_ave)

# Extract public Mg
public_mg_index = [x for x in mg_neighbors_H2 if x in mg_neighbors_H1][0]
print("The public Mg atomic number is ", public_mg_index)
mg_position = positions[public_mg_index - 1]
# Initialize Counter
h_count_bh = 0
h_count_sbh = 0
h_count = 0
cell_multiplier = [-1, 0, 1]  # Multiplier of lattice periodicity
# condition
x_threshold = 3.77
z_threshold_1 = 2.0
z_threshold_2 = 0.6
y_threshold = 2.5

# Traverse atoms to find the number of H atoms that meet the condition
for i, (symbol, position) in enumerate(zip(chemical_symbols, positions)):
    if symbol == 'H':
        for x_multiplier in cell_multiplier:
            for y_multiplier in cell_multiplier:
                for z_multiplier in cell_multiplier:
                    # Considering the periodicity of the lattice, calculate the position of the H atom
                    h_position_relative = np.dot([x_multiplier, y_multiplier, z_multiplier], cell)
                    h_position_absolute = mg_position + h_position_relative

                    # Check if the H atom is within the specified range
                    diff = h_position_absolute - position
                    if abs(diff[0]) < x_threshold and abs(diff[1]) < y_threshold and abs(diff[2]) < z_threshold_1:
                        h_count += 1

print(f"Within the range adjacent to the position of Mg atoms, the number of H atoms is  {h_count}")
for i, (symbol, position) in enumerate(zip(chemical_symbols, positions)):
    if symbol == 'H':
        for x_multiplier in cell_multiplier:
            for y_multiplier in cell_multiplier:
                for z_multiplier in cell_multiplier:
                    h_position_relative = np.dot([x_multiplier, y_multiplier, z_multiplier], cell)
                    h_position_absolute = mg_position + h_position_relative
                    diff = h_position_absolute - position
                    if abs(diff[0]) < x_threshold and abs(diff[1]) < y_threshold and abs(diff[2]) < z_threshold_2:
                        h_count_sbh += 1

k2 = 6 - h_count_sbh
print(f"Within the range adjacent to the position of Mg atoms, the number of sbH atoms is  {h_count_sbh}")
h_count_bh = h_count - h_count_sbh
k1 = 4 - h_count_bh
print(f"Within the range adjacent to the position of Mg atoms, the number of bH atoms is  {h_count_bh}")
# Traverse atoms to find the number of transition metal atoms that meet the conditions
h_count_tm = {}
electronegativity_sum = 0


# Define element serial number and Electronegativity
def is_metal(symbol_metal):
    periodic_table = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12,
        'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'K': 19, 'Ar': 18, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23,
        'Cr': 24, 'Mn': 25, 'Fe': 26, 'Ni': 27, 'Co': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45,
        'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
        'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67,
        'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
        'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95,
        'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
        'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115,
        'Lv': 116, 'Ts': 117, 'Og': 118
    }
    group_num = periodic_table[symbol_metal]
    return 3 <= group_num <= 4 or 11 <= group_num <= 13 or 19 <= group_num <= 32 or 37 <= group_num <= 51 or 55 <= group_num <= 84 or group_num >= 87


def get_electronegativity(atom_type):
    electronegativities = {
        "H": 2.20, "He": 4.40, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
        "Ne": 4.90, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": 3.81,
        "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88,
        "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00,
        "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 1.90, "Ru": 2.20, "Rh": 2.28,
        "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66, "Xe": 2.57,
        "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.20,
        "Gd": 1.20, "Tb": 1.23, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10, "Lu": 1.27, "Hf": 1.30,
        "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62,
        "Pb": 1.87, "Bi": 2.02, "Po": 2.00, "At": 2.20, "Rn": 2.20, "Fr": 0.70, "Ra": 0.90, "Ac": 1.10, "Th": 1.30,
        "Pa": 1.50, "U": 1.38, "Np": 1.36, "Pu": 1.28, "Am": 1.13, "Cm": 1.28, "Bk": 1.30, "Cf": 1.30, "Es": 1.30,
        "Fm": 1.30, "Md": 1.30, "No": 1.30, "Lr": 1.30, "Rf": 1.30, "Db": 1.30, "Sg": 1.30, "Bh": 1.30, "Hs": 1.30,
        "Mt": 1.30, "Ds": 1.30, "Rg": 1.30, "Cn": 1.30, "Nh": 1.30, "Fl": 1.30, "Mc": 1.30, "Lv": 1.30, "Ts": 1.30,
        "Og": 1.30,
    }
    return electronegativities.get(atom_type, None)


for i, (symbol, position) in enumerate(zip(chemical_symbols, positions)):
    if is_metal(symbol):
        for x_multiplier in cell_multiplier:
            for y_multiplier in cell_multiplier:
                for z_multiplier in cell_multiplier:
                    h_position_relative = np.dot([x_multiplier, y_multiplier, z_multiplier], cell)
                    h_position_absolute = mg_position + h_position_relative
                    diff = h_position_absolute - position
                    if abs(diff[0]) < 7.55 and abs(diff[1]) < 1 and abs(diff[2]) < 1:
                        electronegativity = get_electronegativity(symbol)
                        if symbol in h_count_tm:
                            h_count_tm[symbol] += 1
                            electronegativity_sum += electronegativity
                        else:
                            h_count_tm[symbol] = 1
                            electronegativity_sum += electronegativity

print("Metal elements and corresponding quantities：")
for metal, count in h_count_tm.items():
    print(f"{metal}: {count}")

print(f"Multiply and sum the quantity and the corresponding Electronegativity：{electronegativity_sum}")
oh = 1 + 0.125 * k1 - 0.125 * k2
oe = 5 * 1.31 / electronegativity_sum
m = oh * oe * cohp_max_ave * (distance - 0.75)
if k1 < 2:
    Da = 1.83 * (1 + 0.125 * k1 - 0.125 * k2) * 5 * 1.31 / electronegativity_sum * (
        distance - 0.75) * cohp_max_ave - 1.66
else:
    Da = 1.83 * (1 + 0.125 * k1 + 0.125 * k2) * 5 * 1.31 / electronegativity_sum * (
        distance - 0.75) * cohp_max_ave - 1.66
print(f"Calculated value of MgH2 dehydrogenation barrier descriptor：{Da}")
