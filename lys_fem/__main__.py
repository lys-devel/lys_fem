"""
ngsolve main module.
"""

import argparse
parser = argparse.ArgumentParser(prog='lys', usage="mpirun python -m lys_fem (options)", add_help=True)
parser.add_argument("-i", "--input", help="input file", type=str, default="input.dic")
parser.add_argument("-nt", "--numThreads", help="number of threads", type=int, default=4)
args = parser.parse_args()


def run_ngs(file):
    from .fem import FEMProject
    from lys_fem.fem.run import run
    with open(file, "r") as f:
        d = eval(f.read())
    fem = FEMProject(2)
    fem.loadFromDictionary(d)
    run(fem, save=False, nthreads=args.numThreads)


run_ngs(args.input)
