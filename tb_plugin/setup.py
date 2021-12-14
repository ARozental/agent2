import os
import shutil
import pathlib
import setuptools
import subprocess


class BuildFrontend(setuptools.Command):
    """Build the frontend"""
    description = "run npm build on frontend directory using docker"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cwd = pathlib.Path().absolute()
        root = pathlib.Path(__file__).parent.absolute()
        os.chdir(root / "frontend")

        # Run the build
        subprocess.run(['docker-compose', 'up', '-d'], check=True)
        subprocess.run(['docker-compose', 'run', '--rm', 'npm', 'install'], check=True)
        subprocess.run(['docker-compose', 'run', '--rm', 'npm', 'run', 'build:copy'], check=True)
        os.chdir(cwd)  # Restore the working directory


# Specifying setup.py makes a plugin installable via a Python package manager.
# `entry_points` is an important field makes plugins discoverable by TensorBoard
# at runtime.
# See https://packaging.python.org/specifications/entry-points/
setuptools.setup(
    name="agent_plugin",
    version="0.1.0",
    cmdclass={
        "build_fe": BuildFrontend
    },
    description="AGENT Plugin for Tensorboard.",
    packages=["agent_plugin"],
    package_data={
        "agent_plugin": ["static/**"],
    },
    entry_points={
        "tensorboard_plugins": [
            "agent = agent_plugin.plugin:AgentPlugin",
        ],
    },
)
