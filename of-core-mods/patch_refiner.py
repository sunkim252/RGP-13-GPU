#!/usr/bin/env python3
import os, sys

root = os.path.expanduser("~/openfoam/RGP-13/of-core-mods/fvMeshTopoChangers")
cpath = os.path.join(root, "refiner", "refiner_fvMeshTopoChanger.C")
mpath = os.path.join(root, "Make", "files")

# 1) Retarget Make/files LIB to FOAM_USER_LIBBIN
with open(mpath) as f:
    m = f.read()
old_lib = "LIB = $(FOAM_LIBBIN)/libfvMeshTopoChangers"
new_lib = "LIB = $(FOAM_USER_LIBBIN)/libfvMeshTopoChangers"
assert old_lib in m, "Make/files LIB line not found"
m = m.replace(old_lib, new_lib)
with open(mpath, "w") as f:
    f.write(m)
print("Make/files patched -> FOAM_USER_LIBBIN")

# 2) Patch the .C
with open(cpath) as f:
    c = f.read()

# 2a) Add polyDistributionMap.H include (for template distributeCellData)
inc_anchor = '#include "addToRunTimeSelectionTable.H"'
inc_add = inc_anchor + '\n#include "polyDistributionMap.H"'
assert inc_anchor in c, "include anchor not found"
assert '#include "polyDistributionMap.H"' not in c, "include already present"
c = c.replace(inc_anchor, inc_add, 1)

# 2b) Replace the distribute() body
old_block = (
    "void Foam::fvMeshTopoChangers::refiner::distribute\n"
    "(\n"
    "    const polyDistributionMap& map\n"
    ")\n"
    "{\n"
    "    // Redistribute the mesh cutting engine\n"
    "    meshCutter_.distribute(map);\n"
    "}\n"
)
new_block = (
    "void Foam::fvMeshTopoChangers::refiner::distribute\n"
    "(\n"
    "    const polyDistributionMap& map\n"
    ")\n"
    "{\n"
    "    // Redistribute the mesh cutting engine (cellLevel/pointLevel/history)\n"
    "    meshCutter_.distribute(map);\n"
    "\n"
    "    // Redistribute the protected-cell marking to the new decomposition.\n"
    "    // Upstream refiner::distribute only handles meshCutter_; protectedCells_\n"
    "    // is sized to the OLD local cell count and would be stale (wrong size /\n"
    "    // mis-indexed) after redistribution, corrupting the next (un)refinement.\n"
    "    if (protectedCells_.size())\n"
    "    {\n"
    "        labelList protectedCell(protectedCells_.size());\n"
    "        forAll(protectedCell, celli)\n"
    "        {\n"
    "            protectedCell[celli] = protectedCells_.get(celli);\n"
    "        }\n"
    "\n"
    "        map.distributeCellData(protectedCell);\n"
    "\n"
    "        PackedBoolList newProtectedCell(protectedCell.size());\n"
    "        forAll(protectedCell, celli)\n"
    "        {\n"
    "            newProtectedCell.set(celli, protectedCell[celli]);\n"
    "        }\n"
    "        protectedCells_.transfer(newProtectedCell);\n"
    "    }\n"
    "}\n"
)
assert old_block in c, "distribute() block not matched exactly"
c = c.replace(old_block, new_block, 1)

with open(cpath, "w") as f:
    f.write(c)
print("refiner_fvMeshTopoChanger.C patched: include + protectedCells_ redistribution")
