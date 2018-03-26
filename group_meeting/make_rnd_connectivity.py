#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 18:50:45 2018

@author: andrea
"""
import numpy as np
import matplotlib.pyplot as plt


def make_rnd_connectivity(N, density=0.2, connectivity_strength=1.0):
    C = np.exp(np.random.randn(N, N))
    C[np.random.rand(N, N) > density] = 0
    C[np.eye(N, dtype=bool)] = 0
    C *= connectivity_strength * N / C.sum()
    return C
