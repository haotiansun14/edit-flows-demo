from abc import ABC, abstractmethod

import torch
from torch import Tensor
from einops import rearrange


def x2prob(x: Tensor, vocab_size: int) -> Tensor:
    """Converts sequence of tokens to class distribution representation
    """
    return torch.nn.functional.one_hot(x, num_classes=vocab_size).float()


def sample_p(pt: Tensor, temperature: float = 1.0) -> Tensor:
    """Samples protein sequence from class distribution representation
    """
    b, l, _ = pt.shape
    pt = rearrange(pt, 'b l c -> (b l) c')
    xt = torch.multinomial(pt / temperature, 1)
    return xt.reshape(b, l)


class Coupling(ABC):
    @abstractmethod
    def sample(self, x1: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class EmptyCoupling(Coupling):
    """A coupling that samples empty prior sequences."""
    def sample(self, x1: Tensor):
        x0 = torch.empty((x1.shape[0], 0), dtype=x1.dtype, device=x1.device).long()
        return x0, x1


class ExtendedCoupling(Coupling):
    """A coupling that randomly inserts tokens into the target sequence."""
    def sample(self, x1: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class UniformCoupling(Coupling):
    """A coupling that samples uniform prior sequences within a given length range."""
    def __init__(
        self,
        min_len: int = 0,
        max_len: int = 100,
        vocab_size: int = 128,
        mirror_len: bool = False,
        pad_token: int = 129,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.mirror_len = mirror_len
        self.pad_token = pad_token

    def sample(self, x1: Tensor):
        batch_size, _ = x1.shape
        x1_pad_mask = (x1 == self.pad_token)
        if self.mirror_len:
            x0_pad_mask = x1_pad_mask
            x0_max_len = x1.shape[1]
        else:
            x0_seq_len = torch.randint(self.min_len, self.max_len + 1, size=(batch_size,)).long()
            x0_max_len = int(x0_seq_len.max().item())
            x0_pad_mask = torch.arange(x0_max_len, device=x1.device).expand(batch_size, -1) >= x0_seq_len.unsqueeze(1)

        x0 = torch.randint(0, self.vocab_size, size=(batch_size, x0_max_len), dtype=x1.dtype, device=x1.device)
        x0[x0_pad_mask] = self.pad_token
        return x0, x1


class KappaScheduler(ABC):

    @abstractmethod    
    def __call__(self, t: Tensor) ->  Tensor:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, t: Tensor) -> Tensor:
        raise NotImplementedError


class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 2.0, b: float = 0.5) -> None:
        self.a = a
        self.b = b

    def __call__(self, t: Tensor) -> Tensor:
        return -2* (t**3) + 3 * (t**2) + self.a * (t ** 3 - 2* t**2 + t) + self.b * (t**3 - t**2)

    def derivative(self, t: Tensor) -> Tensor:
        return -6 * (t**2) + 6 * t + self.a * (3 * t**2 - 4 * t + 1) + self.b * (3 * t**2 - 2 * t)
