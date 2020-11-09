
class Classes():
    def __init__(self):
        self.defect = 'defect'
        self.non_defect = 'non_defect'

    def toList(self):
        return [self.defect, self.non_defect]

CLASSES = Classes()

class DatasetTypes:
    def __init__(self):
        self.train = 'train'
        self.test = 'test'

    def toList(self):
        return [self.train, self.test]

DATASET_TYPES = DatasetTypes()

class PatchSize():
    def __init__(self, x = 32, y = 32):
        self.x = x
        self.y = y
    def __str__(self):
        return str(self.x)+'x'+str(self.y)


