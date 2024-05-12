"""
ngsolve main module.
"""

import argparse
parser = argparse.ArgumentParser(prog='lys', usage="mpirun python -m lys_fem.mf (options)", add_help=True)
parser.add_argument("-i", "--input", help="input file", type=str, default="input.dic")
args = parser.parse_args()


def run_ngs(file):
    from ..fem import FEMProject
    from .run import run
    with open(file, "r") as f:
        d = eval(f.read())
    fem = FEMProject(2)
    fem.loadFromDictionary(d)
    run(fem, save=False)


run_ngs(args.input)
