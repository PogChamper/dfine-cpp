from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


class PlatformWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform = super().get_tag()
        return "py3", "none", platform

    def run(self) -> None:
        library = Path(__file__).resolve().parent / "dfine" / "libdfine.so"
        if not library.is_file():
            raise RuntimeError(
                "dfine/libdfine.so is not staged; build native wheels with build_wheel.sh"
            )
        super().run()


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": PlatformWheel})
