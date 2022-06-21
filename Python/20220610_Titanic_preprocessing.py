# from sklearn.preprocessing import LabelEncoder
# import importlib


def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown'
    elif age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else: cat = 'Elderly'
    return cat

def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

def drop_features(df):
    df.drop(['PassengerId', 'Ticket', 'Name'], axis=1, inplace=True)
    return df

def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Sex', 'Cabin', 'Embarked']
    for f in features:
        encoder = __import__('sklearn.preprocessing', globals(), locals(), ['LabelEncoder()'], 0).LabelEncoder()
        # encoder = importlib.import_module('sklearn.preprocessing', 'preprocessing').LabelEncoder()

        # encoder = LabelEncoder()
        df[f] = encoder.fit_transform(df[f])
    return df

def preprocessing(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df
