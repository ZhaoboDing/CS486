import numpy as np
import copy


class Factor:
    def __init__(self, vars, probs):
        self.header = np.array(sorted(vars))
        if type(probs) is np.ndarray:
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
        return Factor(new_header, new_table)

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
        return Factor(self.header, self.table)

    def __str__(self):
        table_format = '{:<15}' * (len(self.header) + 1)
        result = [table_format.format(*(self.header.tolist() + ['probability']))]
        for index, value in np.ndenumerate(self.table):
            row = ['True' if boolean > 0 else 'False' for boolean in index]
            result.append(table_format.format(*(row + [value])))

        return '\n'.join(result) + '\n'


def inference(factors, query, hidden_list, evidence_list, print_step=True):
    def step_printer(factor_list):
        for factor in factor_list:
            print(factor)

    factor_list = [copy.copy(factor) for factor in factors]

    if print_step:
        print('Initialized factors:')
        step_printer(factor_list)

    for factor in factor_list:
        for evidence, value in evidence_list.items():
            factor.restrict(evidence, value)
    if print_step:
        print('After restriction:')
        step_printer(factor_list)

    product = np.prod(factor_list)

    if print_step:
        print('After production:')
        step_printer([product])

    for hidden in hidden_list:
        product.sumout(hidden)

    if print_step:
        print('After summing out:')
        step_printer([product])

    product.normalize()

    if print_step:
        print('After normalization:')
        step_printer([product])

    if set(query) == set(product.header):
        return product
    else:
        raise Exception("Invalid query size.")
