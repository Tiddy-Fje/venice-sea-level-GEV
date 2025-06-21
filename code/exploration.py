#%% 
import pyextremes as ext
import matplotlib.pyplot as plt
import pandas as pd
import implementation

dd = (
    pd
    .read_csv('../data/venice90.csv', usecols=['day','month','year', 'sealevel'])
    .astype(float)
)

#%%

date = pd.to_datetime( dd.drop(columns=['sealevel']) )
dd = pd.concat( [date, dd['sealevel']], axis=1, keys=['date', 'sealevel'] )
dd.set_index('date', inplace=True)

dd_series = pd.Series( dd['sealevel'] )
print( dd_series.head() )

#%%

model = ext.EVA(data=dd_series)

model.get_extremes(
    method='BM',
    extremes_type='high',
    block_size='365.2425D',
    errors='raise',
)

fig, ax = plt.subplots(figsize=(8, 6), layout='tight')
model.plot_extremes( ax=ax )
ax.set_ylabel('Sea level [cm]')
ax.set_xlabel('Time of occurrence')
plt.savefig('../figures/data-series.png')

# %%
