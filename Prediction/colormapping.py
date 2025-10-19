# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 22:26:27 2025

@author: 18811
"""

import pandas as pd
import numpy as np
import os
from matplotlib.colors import Normalize, to_hex, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler


def colormap(data, start_site=None, end_site=None):
    if start_site is not None:
        data = data.iloc[start_site:end_site, :]
    
    scaler = MinMaxScaler()
    data['att_score'] = scaler.fit_transform(data['att_score'].values.reshape(-1, 1))
    
    norm = Normalize(vmin=0, vmax=1)
    colors_hex = ['#F4F5F6','#FFF6F4','#FFF4EC','#FFE1D8','#b20012']
    cmap_name = 'my_list'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors_hex, N=len(data))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    data['color'] = data['att_score'].apply(lambda x:to_rgb(sm.to_rgba(x)))
    data.insert(0,'chain','A')

    return data[['chain','resi','color']]
   

path = r"./Data"

protein_repo = pd.read_csv(os.path.join(path, 'protein_repository.tsv'), sep='\t')
gene2seq = {x['Gene Names']:x['Sequence'] for x in protein_repo.to_dict("records")}

prot = 'RAB14'
data = pd.read_csv(os.path.join(path, f'screening_result/{prot}_LigandA_residue_score.csv'))
cols_drop = data.columns[(data == 0).any()]
data = data.drop(columns=cols_drop).T.reset_index(drop=False).astype(np.float64)
data.columns = ['res_pos', 'att_score']
data['res_pos'] += 1
data['res'] = [r for r in gene2seq[prot]][:len(data)]

data = colormap(data)
data.to_csv(os.path.join(path, f'screening_result/{prot}_FFF004.csv'), index=False, encoding="utf-8-sig")
