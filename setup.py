from __future__ import annotations

import os
import sys
import re
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


PLATFORM_NAME_TO_CMAKE_ARCH = {
    'win32': 'Win32',
    'win-amd64': 'x64',
    'win-arm32': 'ARM',
    'win-arm64': 'ARM64'
}


class CMakeGenerator:
    def __init__(self) -> None:
        self._auto_default_generator, self._available_generators = \
            CMakeGenerator._get_available_generators()

        self.name = os.environ.get('CMAKE_GENERATOR', '')
        self.extra_cmake_configure_args: list[str] = []
        if not self.name:
            self.name = self._get_default()
        # If user specified generator - check whenever it is available
        if self.name:
            is_valid_generator_name = any(
                gen.startswith(self.name) for gen in self._available_generators
            )
            if not is_valid_generator_name:
                available_generators_bullet_list = '\n'.join(
                    f' - {generator}'
                    for generator in self._available_generators
                )
                raise ValueError(
                    f'Specified CMAKE_GENERATOR="{self.name}" is not available'
                    ' for this platform. List of available generators:\n'
                    f'{available_generators_bullet_list}'
                )

    @property
    def supports_multi_config(self) -> bool:
        for prefix in ('Ninja Multi-Config', 'Xcode', 'Visual Studio'):
            if self.name.startswith(prefix):
                return True
        return False

    def _get_default(self, ) -> str:
        import importlib.util

        if 'Ninja' in self._available_generators:
            return 'Ninja'
        # ninja can be installed as a wheel
        ninja_module_spec = importlib.util.find_spec('ninja')
        if ninja_module_spec is not None:
            ninja = importlib.util.module_from_spec(ninja_module_spec)
            self.extra_cmake_configure_args.append(
                f'-DCMAKE_MAKE_PROGRAM:FILEPATH={Path(ninja.BIN_DIR)/"ninja"}'
            )
            return 'Ninja'
        return self._auto_default_generator

    @staticmethod
    def _get_available_generators() -> tuple[str, list[str]]:
        cmake_help = subprocess.check_output(('cmake', '--help'),
                                             stderr=subprocess.STDOUT,
                                             universal_newlines=True)
        generators_section = cmake_help[cmake_help.index('Generators'):]

        generator_regex = re.compile(r'([\S\ \-\[\]]+)[\n\ ]*= *')
        # Scan though all matches of generator regex and strip extra spaces
        available_generators: list[str] = []
        default_generator = ''
        for match in re.finditer(generator_regex, generators_section):
            generator = match.group(1).strip()
            # Check if generator is marked as default - it should contain '*'
            # in its name prefix
            is_default = generator.startswith('*')
            generator = generator.strip(' *')

            # Trim possible [arch] hint suffix
            arch_idx = generator.find('[arch]')
            if arch_idx > 0:
                generator = generator[:arch_idx]

            if is_default:
                assert not default_generator, \
                    f'Trying to set "{generator}" as default, but default is'\
                    f' already set to "{default_generator}"'
                default_generator = generator
            available_generators.append(generator)
        assert default_generator, 'Can not find default CMake generator'
        return default_generator, available_generators


def prepare_cmake_configure_args(cmake_generator: CMakeGenerator,
                                 build_type: str,
                                 extension_dir: Path) -> list[str]:
    # Common configure args
    cmake_configure_args = [
        f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_dir}{os.sep}',
        # Set current Python executable to be used for extension build
        f'-DPYTHON_EXECUTABLE={sys.executable}',
        f'-DCMAKE_BUILD_TYPE={build_type}'
    ]

    # Check for extra CMake configure args
    if 'CMAKE_CONFIGURE_EXTRA_ARGS' in os.environ:
        cmake_configure_args.extend((
            arg for arg in os.environ['CMAKE_CONFIGURE_EXTRA_ARGS'].split(' ')
            if arg
        ))

    if sys.platform.startswith('darwin'):
        archs = re.findall(r'-arch (\S+)', os.environ.get('ARCHFLAGS', ''))
        if archs:
            cmake_configure_args.append(
                f'-DCMAKE_OSX_ARCHITECTURES={";".join(archs)}'
            )

    if cmake_generator.supports_multi_config:
        cmake_configure_args.append(
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_'
            f'{build_type.upper()}={extension_dir}'
        )

    cmake_configure_args.append(f'-G{cmake_generator.name}')
    cmake_configure_args.extend(cmake_generator.extra_cmake_configure_args)

    return cmake_configure_args


def prepare_cmake_build_args(cmake_generator: CMakeGenerator,
                             build_type: str) -> list[str]:
    cmake_build_args: list[str] = []

    if cmake_generator.supports_multi_config:
        cmake_build_args.extend(('--config', build_type))

    return cmake_build_args


class CMakeExtension(Extension):
    def __init__(self, name: str,
                 source_dir: Path | os.PathLike | str = '') -> None:
        super().__init__(name, sources=[])
        self.source_dir = Path(os.path.realpath(source_dir))


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        extension_full_path = Path.cwd() / self.get_ext_fullpath(ext.name)
        extension_dir = extension_full_path.parent.resolve()

        cmake_generator = CMakeGenerator()
        build_type = os.environ.get('CMAKE_BUILD_TYPE', 'Release')

        cmake_configure_args = prepare_cmake_configure_args(cmake_generator,
                                                            build_type,
                                                            extension_dir)

        # If MSVC compiler is used and cmake generator name contains arch spec
        # - add it to configure arguments
        if self.compiler.compiler_type == 'msvc' and \
                any(arch in cmake_generator.name for arch in ('ARM', 'Win64')):
            cmake_configure_args.extend(
                ('-A', PLATFORM_NAME_TO_CMAKE_ARCH[self.plat_name])
            )

        cmake_build_args = prepare_cmake_build_args(cmake_generator,
                                                    build_type)
        # Unified way to control the parallelism level during build across
        # all generators. Documentation:
        # https://cmake.org/cmake/help/latest/envvar/CMAKE_BUILD_PARALLEL_LEVEL.html
        if 'CMAKE_BUILD_PARALLEL_LEVEL' not in os.environ:
            if hasattr(self, 'parallel') and self.parallel:
                cmake_build_args.extend(('--parallel', self.parallel))

        build_temp = Path(self.build_temp) / ext.name

        # Run configure
        subprocess.check_call(
            ('cmake', ext.source_dir, '-B', build_temp, *cmake_configure_args)
        )

        # Run build
        subprocess.check_call(
            ('cmake', '--build', build_temp, *cmake_build_args)
        )


if __name__ == '__main__':
    PACKAGE_NAME = 'nope'

    setup(
        name=PACKAGE_NAME,
        ext_modules=[
            CMakeExtension(f'{PACKAGE_NAME}._{PACKAGE_NAME}', './src/native')
        ],
        cmdclass={'build_ext': CMakeBuild},
    )
