from setuptools import setup, find_packages


setup(name='gust',
      version='0.1',
      description='Graph Utilities and STorage',
      url='http://gitlab.lrz.de/kdd-group/gust',
      author='Oleksandr Shchur',
      author_email='shchur@in.tum.de',
      packages=find_packages('.'),
      zip_safe=False)
