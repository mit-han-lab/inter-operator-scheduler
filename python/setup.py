import os
from setuptools import setup
from setuptools import find_packages
import glob

libs = ['../build/libios_runtime.so', '../build/libtrt_runtime.so']
existing_libs = []
for libpath in libs:
    if os.path.exists(libpath):
        existing_libs.append(libpath)

assert any('libios_runtime.so' in lib for lib in existing_libs), "IOS runtime library not found."

setup(name='ios',
      version='0.1.dev0',
      description='IOS: An Inter-Operator Scheduler for CNN Acceleration',
      zip_safe=False,
      packages=find_packages(),
      url='https://github.com/mit-han-lab/inter-operator-scheduler',
      include_package_data=True,
      install_requires=[
            "numpy",
            "pydot",
            "tqdm"
      ],
      data_files=[('ios', existing_libs),
                  ('ios', [*glob.glob('ios/models/randwire_graphs/generated/*'), 'ios/models/randwire_graphs/ws.py'])]
      )

