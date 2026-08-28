from __future__ import annotations
import abc
from pathlib import Path


class SeparationBackend(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    async def separate(self, input_audio: str, output_dir: str) -> tuple[str, str]:
        ...

    async def close(self):
        pass


class SeparationRegistry:
    _backends: dict[str, type] = {}

    @classmethod
    def register(cls, backend_class: type) -> None:
        instance = backend_class()
        cls._backends[instance.name] = backend_class

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._backends:
            raise ValueError(f"Unknown separation backend: {name}")
        return cls._backends[name]

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._backends.keys())

    @classmethod
    async def create(cls, name: str, **kwargs):
        backend_class = cls.get(name)
        return backend_class(**kwargs)


# Register built-in backends
def _register_backends():
    from . import uvr_mdx, htdemucs, bs_roformer
    SeparationRegistry.register(uvr_mdx.UVRMdxBackend)
    SeparationRegistry.register(htdemucs.HTDemucsBackend)
    SeparationRegistry.register(bs_roformer.BSRoFormerBackend)

_register_backends()