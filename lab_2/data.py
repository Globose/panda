import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

FILES = ['lab_2/1-titanic-small.csv', 'lab_2/2-titanic2attr.csv', 'lab_2/3-titanic.csv']


def load_csv(path):
    """Loads csv file"""
    dataFrame = pd.read_csv(path, sep=",", encoding="utf-8")
    return dataFrame


def normalize(df, limit=1):
    "Normalizes data in dataframe"
    scaler = MinMaxScaler(feature_range=(0,limit))
    scaler.fit(df)
    df_2 = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
    return df_2


def encode_labels(df):
    "Encodes labels"
    labels = ['Ticket', 'Sex', 'Embarked', 'Fare']
    le = LabelEncoder()
    for l in labels:
        df[l] = le.fit_transform(df[l])
    return df


def df_titanic_properties():
    df = load_csv(FILES[2]).dropna()
    df = encode_labels(df)
    
    print("Shape \n", df.shape, "\n")
    print("dtypes \n", df.dtypes, "\n")

    df_fem  = df[df['Sex']==0]
    df_male  = df[df['Sex']==1]
    
    print("Female passengers: ", len(df_fem))
    print("Survived: ", df_fem['Survived'].sum())
    print("Percentage: ", df_fem['Survived'].sum()/len(df_fem), "\n")
    
    print("Male passengers: ", len(df_male))
    print("Survived: ", df_male['Survived'].sum())
    print("Percentage: ", df_male['Survived'].sum()/len(df_male), "\n")