from distutils.command.install import install
from distutils.command.clean import clean
from distutils.util import convert_path
import setuptools
import distutils
import os
import glob
import shutil


# See: https://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package/2073599#2073599
package_name = "tinygrad_lightning"

main_ns = {}
ver_path = convert_path('%s/version.py' % package_name)
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)


with open("requirements.txt", 'r') as h:
    requirements = [r.replace("\n", "") for r in h.readlines()]

here = os.path.dirname(__name__)


class CleanCommand(clean):
    """Custom clean command to tidy up the project root."""
    CLEAN_FILES = './build ./dist ./.eggs ./*.pyc ./*.tgz ./*.egg-info'.split(
        ' ')

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        global here

        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(
                os.path.join(here, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(here):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError(
                        "%s is not a path inside %s" % (path, here))
                print('removing %s' % os.path.relpath(path))
                shutil.rmtree(path)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=package_name,
    version=main_ns['__version__'],
    description='high level interface for tinygrad',
    author='Malte Koch',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["api", "order"],
    author_email='malte-koch@gmx.net',
    maintainer='Malte Koch',
    maintainer_email='malte-koch@gmx.net',
    url="https://github.com/baudcode/tinygrad-lightning",
    cmdclass={"clean": CleanCommand},
    # namespace_packages=[package_name],
    packages=setuptools.find_packages(include=package_name + "/*"),
    include_data_packages=True,
    # packages=setuptools.find_namespace_packages(exclude=['tests', 'tests.*', "experimental", "experimantal/*"]),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
        ],
    },
    extras_require={},
    zip_safe=False,
    ext_modules=[],
    setup_requires=[],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
