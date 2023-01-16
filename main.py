# Import Necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
# from sklearn.impute import IterativeImputer
# from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Hyperparameters(object):
    """
    The parameters
    """
    filepath: str = "train.csv"
    max_depth: int = 4, 
    max_leaf_nodes: int = 6, 
    min_samples_split: int = 20, 
    n_estimators: int = 350, 
    random_state: int = 34,

# Instantiate Hyperparameters class
hp = Hyperparameters()

# Load the df and the test datasets
def create_dataframe(filepath):
    return pd.read_csv(filepath)

# Group the family_size column
def assign_passenger_label(family_size):
    if family_size == 0:
        return "Alone"
    elif family_size <=3:
        return "Small_family"
    else:
        return "Big_family"

# Group the Ticket column
def assign_label_ticket(first):
    if first in ["F", "1", "P", "9"]:
        return "Ticket_high"
    elif first in ["S", "C", "2"]:
        return "Ticket_middle"
    else:
        return "Ticket_low"
    
# Group the Title column    
def assign_label_title(title):
    if title in ["the Countess", "Mlle", "Lady", "Ms", "Sir", "Mme", "Mrs", "Miss", "Master"]:
        return "Title_high"
    elif title in ["Major", "Col", "Dr"]:
        return "Title_middle"
    else:
        return "Title_low"
    
# Group the Cabin column  
def assign_label_cabin(cabin):
    if cabin in ["D", "E", "B", "F", "C"]:
        return "Cabin_high"
    elif cabin in ["G", "A"]:
        return "Cabin_middle"
    else:
        return "Cabin_low"

# Preprocess dataframe
def preprocess_dataframe(df):
    # Imputers
    imp_embarked = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imp_age = SimpleImputer(strategy="mean")
    # Impute Embarked
    df["Embarked"] = imp_embarked.fit_transform(df[["Embarked"]])
    # Impute Age
    df["Age"] = np.round(imp_age.fit_transform(df[["Age"]]))
    # Initialize a Label Encoder
    le = LabelEncoder()
    # Encode Sex
    df["Sex"] = le.fit_transform(df[["Sex"]].values.ravel())
    # Family Size
    df["Fsize"] = df["SibSp"] + df["Parch"]
    # Ticket first letters
    df["Ticket"] = df["Ticket"].apply(lambda x: str(x)[0])
    # Cabin first letters
    df["Cabin"] = df["Cabin"].apply(lambda x: str(x)[0])
    # Titles
    df["Title"] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

    # Family size
    df["Fsize"] = df["Fsize"].apply(assign_passenger_label)
    # Ticket
    df["Ticket"] = df["Ticket"].apply(assign_label_ticket)
    # Title
    df["Title"] = df["Title"].apply(assign_label_title)
    # Cabin
    df["Cabin"] = df["Cabin"].apply(assign_label_cabin)
    df = pd.get_dummies(columns=["Pclass", "Embarked", "Ticket", "Cabin","Title", "Fsize"], data=df, drop_first=True)
    df.drop(["SibSp", "Parch", "Name", "PassengerId"], axis=1, inplace=True)

    return df


# Select the features and the target
def create_feature_target(df):
    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]
    return X,y

# Fit model
def fit_model(X, y, max_depth, max_leaf_nodes, min_samples_split, n_estimators, random_state):
    # Initialize a RandomForestClassifier
    rf = RandomForestClassifier(max_depth=4, max_leaf_nodes=6, min_samples_split=20, n_estimators=350, random_state=34)
    print("Model is fitted")
    return rf.fit(X, y)

# Run workflow
def run_wf(filepath: str, max_depth: int, max_leaf_nodes: int, min_samples_split: int, n_estimators: int, random_state: int) -> RandomForestClassifier:
    df = create_dataframe(filepath)
    df = preprocess_dataframe(df)
    X,y = create_feature_target(df)
    return fit_model(X=X, y=y, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split, n_estimators=n_estimators, random_state=random_state)

if __name__=="__main__":
    run_wf(hp.filepath, hp.max_depth, hp.max_leaf_nodes, hp.min_samples_split, hp.n_estimators, hp.random_state)


