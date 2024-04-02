"""
This file contains the defition of the base Dataset class.
"""

class DatasetRegistration(type):
    """
    Metaclass for registering different datasets
    """
    def __init__(cls, name, bases, nmspc):
        super().__init__(name, bases, nmspc)
        if not hasattr(cls, 'registry'):
            cls.registry = dict()
        cls.registry[name] = cls

    # Metamethods, called on class objects:
    def __iter__(cls):
        return iter(cls.registry)

    def __str__(cls):
        return str(cls.registry)

class Dataset(metaclass=DatasetRegistration):
    """
    Base Dataset class
    """
    def __init__(self, *args, **kwargs):
        pass