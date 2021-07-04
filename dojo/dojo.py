import abc
from abc import ABC, abstractmethod

class Dojo(ABC):
    """
    The dojo is where models are trained/tested for a specific purpose (objective function) in a specific way (training algorithm).

    The specific training objective and algorithm can be tested for different models, dataloaders, optimizers, and criteria.
    """

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

    @abstractmethod
    def obj_func(self, *args):
        """
        Objective function should be implemented here.

        :param *args: list of arguments
        :return: Loss/Value/etc.
        """
        pass


