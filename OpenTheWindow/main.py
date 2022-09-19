import numpy as np
naaaaaaaaaah
git gud
import matplotlib as plt




class Dog:
    def __init__(self):
        self.name = None
        self.age = None
        self.race = None

    def set_name(self, name):
        self.name = name#

    def set_age(self, age):
        self.age = age

    def print_name(self):
        print(self.name)

    def __len__(self):
        return self.age

    def __add__(self, other):
        return self.age+other.age


eugenios_dog = Dog()
eugenios_dog.set_age(12)
nils_dog = Dog()
nils_dog.set_age(8)
print(nils_dog.__add__(nils_dog))
print(eugenios_dog+nils_dog)
b = 5
b = int("5", base=16)




