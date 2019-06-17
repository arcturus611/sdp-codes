#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 17:43:34 2019

@author: arcturus
"""

import numpy as np

def generate_psd(s):
    A = np.random.rand(s)
    Asym = (1/2)*(A + A.transpose())
    Apsd = Asym + s*np.eye(s)
    return Apsd
    