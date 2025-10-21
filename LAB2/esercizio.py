import re
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sea
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.expand_frame_repr', False)
path = os.path.join('datasets', 'train.csv')
df = pd.read_csv(path)
"""print(df.head())
sea.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()
sea.catplot(x='Survived', col='Sex', kind='count', data=df)
plt.show()
sea.countplot(x='Survived', hue='Pclass', data=df)
plt.show()
sea.countplot(x='SibSp', data=df)
plt.show()
sea.histplot(df['Age'].dropna())
plt.show()
sea.boxplot(x='Pclass', y='Age', data=df, palette='winter') #boxplot per visualizzare la distribuzione dell'età in base alla classe
#i punti fuori dal boxplot rappresentano i valori anomali, high leverage points
plt.show()"""
#passangerId, Embarked e Ticket non sono rilevanti per l'analisi, Name lo teniamo per estrarre il titolo (o Job)
df.drop(labels=['PassengerId', 'Embarked', 'Ticket'], axis=1, inplace=True)

def impute(cols):
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        if pclass == 1:
            return 38 #età media per la prima classe
        elif pclass == 2:
            return 29 #età media per la seconda classe
        else:
            return 24 #età media per la terza classe
    else:
        return age


df['Age'] = df[['Age', 'Pclass']].apply(impute, axis=1) #axis=1 per applicare la funzione sulle righe
#sea.heatmap(df.isnull(), cbar=False, cmap='viridis')
#plt.show()
#se cabin è null, significa che il passeggero non aveva una cabina assegnata, quindi possiamo sostituire i valori null con '0' (No Cabin)
df['Has_Cabin'] = ~df['Cabin'].isnull() #creiamo una nuova colonna che indica se il passeggero aveva una cabina o no
df.drop('Cabin', axis=1, inplace=True) #rimuoviamo la colonna Cabin
#estraiamo il titolo dal nome del passeggero
df['Title'] = df['Name'].apply(lambda x: re.search('([A-Z][a-z]+)\.', x).group(1))

#raggruppiamo i titoli meno comuni in una categoria 'Other'
df['Title'] = df['Title'].replace({
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Miss',
    'Mrs' :'Miss',
})
df['Title']= df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Jonkheer', 'Don', 'Sir', 'Lady', 'Capt'], 'Special')
df.drop('Name', axis=1, inplace=True) #rimuoviamo la colonna Name
sea.countplot(x='Title', data=df)
plt.show()
df['CatAge'] = pd.qcut(df['Age'], 4, labels=False)
df['CatFare'] = pd.qcut(df['Fare'], 4, labels=False)#creiamo una colonna categoriale per l'età
df.drop(['Age', 'Fare'], axis=1, inplace=True) #rimuoviamo la colonna Age e Fare
labelencoder = LabelEncoder()
df['Sex'] = labelencoder.fit_transform(df['Sex'])
df = pd.get_dummies(df, drop_first=True) #convertiamo le colonne categoriali in variabili dummy per non creare ordinalità
print(df.head())

#corr_matrix = df.corr()
#sea.heatmap(corr_matrix, annot=True)
#plt.show()
#possiamo rimuovere title_miss perché è ridondante
df.drop('Title_Miss', axis=1, inplace=True)
corr_matrix = df.corr()
sea.heatmap(corr_matrix, annot=True)
plt.show()
#non c'è bisogno di normalizzare

