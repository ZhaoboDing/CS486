import numpy as np
import copy

class Factor:
    def __init__(self, vars, probs, flatten=False):
        self.header = np.array(sorted(vars))
        if flatten:
            self.table = probs
        else:
            order = np.argsort(vars)
            probs.sort(key=lambda x: tuple(x[i] for i in order))
            self.table = np.array([p[-1] for p in probs])
            self.table = self.table.reshape((2, ) * len(vars))

    def restrict(self, var, val):
        if var not in self.header:
            return

        index = 1 if val else 0
        axis = np.where(self.header == var)[0][0]
        self.header = np.delete(self.header, axis)
        self.table = np.take(self.table, index, axis=axis)

    def __mul__(self, other):
        new_header = np.array(sorted(np.union1d(self.header, other.header)))
        a = self.table.copy()
        b = other.table.copy()

        for index, var in enumerate(new_header):
            if var not in self.header:
                a = np.expand_dims(a, axis=index)
            if var not in other.header:
                b = np.expand_dims(b, axis=index)

        new_table = a * b
        return Factor(new_header, new_table, True)

    def sumout(self, var):
        if var not in self.header:
            return

        axis = np.where(self.header == var)[0][0]
        self.header = np.delete(self.header, axis)
        self.table = np.sum(self.table, axis=axis)

    def normalize(self):
        self.table /= np.sum(self.table)

    def possibility(self, vars):
        if len(vars) != len(self.header):
            raise Exception("Invalid query size.")

        index = []
        for var in self.header:
            if var in vars:
                index.append(1)
            elif "not " + var in vars:
                index.append(0)
            else:
                raise Exception("Invalid query.")

        return self.table.item(tuple(index))

    def __copy__(self):
        return Factor(self.header, self.table, True)

def inference(factors, query, hidden_list, evidence_list):
    factor_list = [copy.copy(factor) for factor in factors]
    for factor in factor_list:
        for evidence, value in evidence_list.items():
            factor.restrict(evidence, value)

    product = np.prod(factor_list)

    for hidden in hidden_list:
        product.sumout(hidden)

    product.normalize()
    if set(query) == set(product.header):
        return product
    else:
        raise Exception("Invalid query size.")
