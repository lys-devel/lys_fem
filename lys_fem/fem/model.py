
models = {}


class FEMModel:
    def saveAsDictionary(self):
        return {"model": self.name}

    @staticmethod
    def loadFromDictionary(d):
        cls_list = set(sum(models.values(), []))
        cls_dict = {m.name: m for m in cls_list}
        model = cls_dict[d["model"]]
        del d["model"]
        return model.loadFromDictionary(d)
