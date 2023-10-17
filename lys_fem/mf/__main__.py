"""
mfem main module.
"""

import argparse
from lys_fem import mf

parser = argparse.ArgumentParser(prog='lys', usage="mpirun python -m lys_fem.mf (options)", add_help=True)
parser.add_argument("-i", "--input", help="input file", type=str, default="input.dic")
parser.add_argument("--mpi", help="Run in parallel", type=bool, default=False)
args = parser.parse_args()

mf.parallel = args.mpi


def run_mfem(file):
    from ..fem import FEMProject
    from .run import run
    with open(file, "r") as f:
        d = eval(f.read())
    fem = FEMProject(2)
    fem.loadFromDictionary(d)
    run(fem)


run_mfem(args.input)
