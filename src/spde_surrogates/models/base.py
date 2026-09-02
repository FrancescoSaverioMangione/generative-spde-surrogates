from abc import ABC, abstractmethod
from torch import nn


class ConditionalGenerator(nn.Module, ABC):
    """
    Common interface for conditional generative models.
    """

    @abstractmethod
    def sample(
        self,
        cond,
        z=None,
        **kwargs,
    ):
        """
        Generate samples conditioned on cond.
        """
        raise NotImplementedError