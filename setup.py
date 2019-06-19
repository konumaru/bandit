# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

import os
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

def strip_comments(l):
    return l.split('#', 1)[0].strip()

def reqs(*f):
    return list(filter( None, [strip_comments(l) for l in open(os.path.join(os.getcwd(), *f)).readlines()] ))

setup(
    name='contextual_bandit',
    version='0.1.0',
    description='A library of contextual multi-arm bandits',
    long_description=readme,
    author='Kenneth Reitz',
    author_email='me@kennethreitz.com',
    url='https://github.com/kennethreitz/samplemod',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=reqs('requirements.txt')
)

