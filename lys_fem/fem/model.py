
models = {}


class FEMModel:
    def saveAsDictionary(self):
        model = None
        for key, m in models.items():
            if isinstance(self, m):
                model = key
                continue
        return {"model": model}

    @staticmethod
    def loadFromDictionary(d):
        model = models[d["model"]]
        del d["model"]
        return model.loadFromDictionary(d)
