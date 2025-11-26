import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

sns.set_palette("husl")
sns.set_style('white')

# ## Loading data

train_df = pd.read_csv("../train_V2.csv")

print("Size of train dataset : " + str(len(train_df)))

print(train_df.head())

features = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'kills', 'killStreaks',
            'longestKill', 'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired']
infos = ['matchDuration', 'matchType', 'maxPlace', 'numGroups']
ELO = ['rankPoints', 'killPoints', 'winPoints']
label = ['winPlacePerc']

#TODO