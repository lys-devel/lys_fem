from . import mfem_orig

if mfem_orig.isParallel():
    SparseMatrix = mfem_orig.HypreParMatrix
else:
    SparseMatrix = mfem_orig.SparseMatrix


class MFEMVector(mfem_orig.Vector):
    def __add__(self, value):
        res = MFEMVector(value.Size())
        mfem_orig.add_vector(self, 1, value, res)
        return res

    def __sub__(self, value):
        res = MFEMVector(self)
        res -= value
        return res

    def __mul__(self, value):
        res = MFEMVector(self)
        res *= value
        return res

    def __str__(self):
        return str([x for x in self])
    
    def __len__(self):
        return len([x for x in self])
    

class MFEMBlockVector(mfem_orig.BlockVector):
    def __add__(self, value):
        res = MFEMBlockVector(self)
        res += value
        return res

    def __sub__(self, value):
        res = MFEMBlockVector(self)
        res -= value
        return res

    def __mul__(self, value):
        res = MFEMBlockVector(self)
        res *= value
        return res

    def __str__(self):
        return str([x for x in self])
    
    def __len__(self):
        return len([x for x in self])


class MFEMMatrix(SparseMatrix):
    def __add__(self, value):
        return MFEMMatrix(mfem_orig.Add(1.0, self,1.0, value))

    def __sub__(self, value):
        return MFEMMatrix(mfem_orig.Add(1.0, self,-1.0, value))

    def __neg__(self):
        res = MFEMMatrix(self)
        res *= -1
        return res

    def __mul__(self, value):
        if isinstance(value, MFEMVector):
            res = MFEMVector(value)
            self.Mult(value, res)
            return res
        else:
            res = MFEMMatrix(self)
            res *= value
            return res
        
    def __rmul__(self, value):
        return self*value


class MFEMBlockOperator(mfem_orig.BlockOperator):
    def __add__(self, value):
        res = MFEMBlockOperator(self.RowOffsets(), self.ColOffsets())
        for r in range(self.NumRowBlocks()):
            for c in range(self.NumColBlocks()):
                if self.IsZeroBlock(r,c):
                    if not value.IsZeroBlock(r,c):
                        res.SetBlock(r,c, value.GetBlock(r,c))
                else:
                    if value.IsZeroBlock(r,c):
                        res.SetBlock(r,c,self.GetBlock(r,c))
                    else:
                        res.SetBlock(r,c,AddOperator(self.GetBlock(r,c), value.GetBlock(r,c)))
        return res

    def __mul__(self, value):
        res = MFEMBlockOperator(self.RowOffsets(), self.ColOffsets())
        for r in range(self.NumRowBlocks()):
            for c in range(self.NumColBlocks()):
                if not self.IsZeroBlock(r,c):
                    res.SetBlock(r,c,MulOperator(self.GetBlock(r,c), value))
        return res
    

class AddOperator(mfem_orig.PyOperator):
    def __init__(self, op1, op2):
        super().__init__(op1.Height(), op1.Width())
        self._op1 = op1
        self._op2 = op2

    def Mult(self, x, y):
        self._op1.Mult(x,y)
        self._op2.AddMult(x,y)


class MulOperator(mfem_orig.PyOperator):
    def __init__(self, op, value):
        super().__init__(op.Height(), op.Width())
        self._op = op
        self._value = value

    def Mult(self, x, y):
        self._op.Mult(x,y)
        y *= self._value

    def MultTranspose(self, x, y):
        self._op.MultTranspose(x,y)
        y *= self._value