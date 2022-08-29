import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import copy

class Test:
    def __int__(self):
        pass

    def task_one(self, param1, param2, param3):
        return param1+param2+param3

    def task_four(self, param: int):
        """
        This function is going to do blablabla
        :param param:
        :return:
        """
        return [param, param, param]

my_dictionary = {"eugenio": "italian", "nils": "german"}
print(my_dictionary["eugenio"])
names = ["wt1", "wt2", "wt3"]
weights = [100, 200, 300]
combine = {wt_name: wt_weight for wt_name, wt_weight in zip(names, weights)}
copied_combine = copy(combine)
print(combine)
for name, weight in zip(names[1:3], weights[1:3]):
    print(name)
    print(weight)
print(combine["wt2"])
