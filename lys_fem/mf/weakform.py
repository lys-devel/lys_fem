from . import mfem, util


class TrialFunction:
    def __init__(self, mesh, ess_tdof, order=1):
        fec =mfem.H1_FECollection(order, mesh.Dimension())
        self._space = mfem.FiniteElementSpace(mesh, fec, 1)
        self._ess_tdof = ess_tdof

    @property
    def space(self):
        return self._space
    
    @property
    def ess_tdof(self):
        return self._ess_tdof


class TestFunction:
    def __init__(self, trial):
        self._trial = trial

    @property
    def space(self):
        return self._trial.space
    
    @property
    def ess_tdof(self):
        return self._trial.ess_tdof


class Coefficient:
    def __init__(self):
        pass


class WeakForm:
    def __init__(self, coef, trial, test):
        self._trial = trial
        self._test = test
        self._coef = coef

    def matrix(self):
        if self._trial.space == self._test.space:
            return util.bilinearForm(self._trial.space, self._trial.ess_tdof, self._integrator())
        else:
            return util.mixedBilinearForm(self._trial.space, self._test.space, self._trial.ess_tdof, self._test.ess_tdof, self._integrator())

    def _integrator(self):
        if isinstance(self._trial, TrialFunction):
            if isinstance(self._test, TestFunction):
                return mfem.MixedScalarMassIntegrator(self._coef)
            else:
                return mfem.MixedScalarWeakDivergenceIntegrator(self._coef)
        else:
            if isinstance(self._test, TestFunction):
                return mfem.MixedDirectionalDerivativeIntegrator(self._coef)
            else:
                return mfem.MixedGradGradIntegrator(self._coef)
