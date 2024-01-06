#!/usr/bin/env python

import os
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))
readme_content = open(os.path.join(here, 'README.md')).read()


exclude_dirs = ['ez_setup', 'test', 'build']


def find_version(basedir):
    with open(os.path.join(basedir, 'version')) as f:
        return f.readline()


def parse_requirements(filename):
    lines = [line.strip() for line in open(filename)]
    return [line for line in lines if line and not line.startswith('#')]


def get_requirements(requirements='requirements.txt'):
    install_reqs = parse_requirements(os.path.join(here, requirements))
    return install_reqs


setup(name='transformer_models',
      description='A library to learn about and play with the internals of a transformer.',
      version=find_version(here),
      long_description=readme_content,
      author='Arnav Gulati',
      author_email='aarnorox@gmail.com',
      url='',
      include_package_data=True,
      install_requires=get_requirements(),
      dependency_links=[],
      packages=find_packages())
