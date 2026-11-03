import os
from glob import glob

from setuptools import setup

package_name = "picklabel_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "weights"), glob("weights/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mohd Waqar Azim",
    maintainer_email="waqarworks11@gmail.com",
    description="PickLabel student detector (drop-in for sam3_ros), pick logger and replay nodes",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detector_node = picklabel_ros.detector_node:main",
            "pick_logger_node = picklabel_ros.pick_logger_node:main",
            "replay_node = picklabel_ros.replay_node:main",
        ],
    },
)
