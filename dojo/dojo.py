import abc
from abc import ABC, abstractmethod

class Dojo(ABC):
    """
    The dojo is where models are trained/tested for a specific purpose (objective function) in a specific way (training algorithm).

    The specific training objective and algorithm can be tested for different models, dataloaders, optimizers, and criteria.
    """

    def __init__(self, obj_func, desc=None):
        """
        Initializes the dojo with an objective function (obj_func) and a description (desc).

        :param obj_func: objective function of this Dojo (What is the model trying to minimize/maximize?)
        :param desc: description of this Dojo
        """
        self.obj_func = obj_func
        self.desc = desc

    @abstractmethod
    def train(self, model, dataloader, optimizer, criteria, logger):
        """
        Training algorithm should be implemented here.

        :param model: model to be trained
        :param dataloader: data loader for training
        :param optimizer: optimizer for training
        :param criteria: stopping criteria for training
        :param logger: the logger logs any importatnt information during training
        :return: model
        """
        pass

    @abstractmethod
    def test(self, model, dataloader, logger):
        """
        Testing algorithm should be implemented here.

        :param model: model to be tested
        :param dataloader: data loader for testing
        :param logger: the logger logs any importatnt information during testing
        :return: average testing metric => loss/accuracy
        """
        pass

