# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 22:25:44 2025

@author: 18811
"""

import csv
import os

path = r"./Data/screening_result"
with open(os.path.join(path, 'RAB14_FFF004.csv.csv'), encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for i,row in enumerate(reader):
        cmd.set_color(f'color_{i}', row["color"])
        cmd.color(f'color_{i}', f"chain {row['chain']} and resi {row['resi']}")