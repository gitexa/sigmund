import os

import pandas as pd
import pyreadstat

path_metadata = os.path.join('data', 'CBCT_alexander.sav')
path_metadata_out = os.path.join('data', 'CBCT_alexander.csv')
df, meta = pyreadstat.read_sav(path_metadata)
df.to_csv(path_metadata_out)

