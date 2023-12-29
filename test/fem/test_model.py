from lys_fem.fem import FEMFixedModel, Equation, DomainCondition, BoundaryCondition, InitialCondition

from ..base import FEMTestCase

class model_test(FEMTestCase):
    def test_equation(self):
        e1 = _Equation(geometries=[1])
        self.assertEqual(e1.geometries.getSelection(), [1])
        self.assertEqual(e1.geometries.geometryType, "Domain")
        self.assertEqual(e1.variableName, "x")

        d = e1.saveAsDictionary()
        e2 = _Equation.loadFromDictionary(d)

        self.assertEqual(e2.geometries.getSelection(), [1])
        self.assertEqual(e2.geometries.geometryType, "Domain")
        self.assertEqual(e2.variableName, "x")

    def test_equations(self):
        model = _Model()

        c1 = _Equation(geometries=[1])
        model.equations.append(c1)

        self.assertEqual(model.equations[1].geometries.getSelection(), [1])
        self.assertEqual(model.equations[1].objName, "Test Equation1")

        d = model.equations.saveAsDictionary()
        d_list = model.equations.loadFromDictionary(d, model.equationTypes)
        self.assertEqual(d_list[1].geometries.getSelection(), [1])

    def test_initialCondition(self):
        c1 = InitialCondition(values=1, geometries=[1])

        self.assertEqual(c1.values, 1)
        self.assertEqual(c1.geometries.getSelection(), [1])
        self.assertEqual(c1.geometries.geometryType, "Domain")

        d = c1.saveAsDictionary()
        c2 = InitialCondition.loadFromDictionary(d)

        self.assertEqual(c1.values, c2.values)
        self.assertEqual(c1.geometries.getSelection(), c2.geometries.getSelection())

    def test_InitialConditions(self):
        model = _Model()

        c1 = InitialCondition(values=1, geometries=[1])
        model.initialConditions.append(c1)

        self.assertEqual(model.initialConditions[0].values, 1)
        self.assertEqual(model.initialConditions[0].geometries.getSelection(), [1])
        self.assertEqual(model.initialConditions[0].objName, "Initial Condition1")

        d = model.initialConditions.saveAsDictionary()
        d_list = model.initialConditions.loadFromDictionary(d, model.initialConditionTypes)
        self.assertEqual(d_list[0].values, 1)

    def test_domainCondition(self):
        c1 = _DomainCondition(values=1, geometries=[1])

        self.assertEqual(c1.values, 1)
        self.assertEqual(c1.geometries.getSelection(), [1])
        self.assertEqual(c1.geometries.geometryType, "Domain")

        d = c1.saveAsDictionary()
        c2 = _DomainCondition.loadFromDictionary(d)

        self.assertEqual(c1.values, c2.values)
        self.assertEqual(c1.geometries.getSelection(), c2.geometries.getSelection())

    def test_domainConditions(self):
        model = _Model()

        c1 = _DomainCondition(values=1, geometries=[1])
        model.domainConditions.append(c1)

        self.assertEqual(model.domainConditions[0].values, 1)
        self.assertEqual(model.domainConditions[0].geometries.getSelection(), [1])
        self.assertEqual(model.domainConditions[0].objName, "Domain Condition Test1")

        d = model.domainConditions.saveAsDictionary()
        d_list = model.domainConditions.loadFromDictionary(d, model.domainConditionTypes)
        self.assertEqual(d_list[0].values, 1)

    def test_boundaryCondition(self):
        c1 = _BoundaryCondition(values=1, geometries=[1])

        self.assertEqual(c1.values, 1)
        self.assertEqual(c1.geometries.getSelection(), [1])
        self.assertEqual(c1.geometries.geometryType, "Boundary")

        d = c1.saveAsDictionary()
        c2 = _DomainCondition.loadFromDictionary(d)

        self.assertEqual(c1.values, c2.values)
        self.assertEqual(c1.geometries.getSelection(), c2.geometries.getSelection())

    def test_boundaryConditions(self):
        model = _Model()

        c1 = _BoundaryCondition(values=1, geometries=[1])
        model.boundaryConditions.append(c1)

        self.assertEqual(model.boundaryConditions[0].values, 1)
        self.assertEqual(model.boundaryConditions[0].geometries.getSelection(), [1])
        self.assertEqual(model.boundaryConditions[0].objName, "Boundary Condition Test1")

        d = model.boundaryConditions.saveAsDictionary()
        d_list = model.boundaryConditions.loadFromDictionary(d, model.boundaryConditionTypes)
        self.assertEqual(d_list[0].values, 1)

    def test_model(self):
        model = _Model()

        c1 = _Equation(geometries=[1])
        model.equations.append(c1)

        b1 = _BoundaryCondition(values=1, geometries=[1])
        model.boundaryConditions.append(b1)

        c1 = _DomainCondition(values=1, geometries=[1])
        model.domainConditions.append(c1)

        i1 = InitialCondition(values=1, geometries=[1])
        model.initialConditions.append(i1)

        d = model.saveAsDictionary()
        m = _Model.loadFromDictionary(d)

        self.assertEqual(m.equations[1].geometries.getSelection(), [1])
        self.assertEqual(m.boundaryConditions[0].geometries.getSelection(), [1])
        self.assertEqual(m.domainConditions[0].geometries.getSelection(), [1])
        self.assertEqual(m.initialConditions[0].geometries.getSelection(), [1])

class _Model(FEMFixedModel):
    className = "Test Model"

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    @classmethod
    @property
    def equationTypes(self):
        return [_Equation]

    @classmethod
    @property
    def domainConditionTypes(self):
        return [_DomainCondition]
    
    @classmethod
    @property
    def boundaryConditionTypes(self):
        return [_BoundaryCondition]

class _Equation(Equation):
    className = "Test Equation"
    def __init__(self, varName="x", **kwargs):
        super().__init__(varName, **kwargs)

        
class _DomainCondition(DomainCondition):
    className = "Domain Condition Test"


class _BoundaryCondition(BoundaryCondition):
    className = "Boundary Condition Test"
