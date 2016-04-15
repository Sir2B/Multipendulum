#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
import py2exe

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.animation as animation

setup(console=['multipendulum.py'])