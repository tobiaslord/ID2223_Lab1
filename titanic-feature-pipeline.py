import os
import modal

LOCAL=True
BACKFILL = False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def name_generator():
    import random
    import string
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(20))

def generate_titanicee(survived, ticket_class, sex, port, fare, age, sibsp, parch):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({
        "survived": [survived],
        "pclass": [random.choice(ticket_class)],
        "sex": [random.choice(sex)],
        "embarked": [random.choice(port)],
        "fare": [random.uniform(fare[0], fare[1])],
        "age": [random.uniform(age[0], age[1])],
        "sibsp": [random.choice(sibsp)],
        "parch": [random.choice(parch)],
        "name": [name_generator()]
        # "syntethic": [True]
    })

    df["age"] = df["age"].astype(str).astype(float)
    df["pclass"] = df["pclass"].astype(str)
    df["embarked"] = df["embarked"].fillna("?")
    df["survived"] = df["survived"].astype(bool)

    print(df.info())
    return df


def get_random_titanicee():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    survived = generate_titanicee(survived = True,
        ticket_class=[1, 2, 3],
        sex=["male", "female", "female"],
        port=["S", "C", "Q"],
        fare=[20, 150],
        age=[0, 50],
        sibsp=[0, 0, 1, 1, 1, 2, 2, 3],
        parch=[0, 1, 1, 1, 2, 3])

    died = generate_titanicee(survived = False,
        ticket_class=[1, 2, 2, 3, 3, 3, 3, 3, 3, 3],
        sex=["male", "male", "female"],
        port=["S", "C", "Q"],
        fare=[0, 50],
        age=[0, 50],
        sibsp=[0, 0, 1, 1, 2, 2, 3, 4, 5, 6],
        parch=[0, 0, 1, 2, 2, 2, 3, 4, 5, 6, 7])


    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        df = survived
        print("Survivor added")
    else:
        df = died
        print("Dead person added")

    print(df)

    return df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()


    primary_keys = ["pclass", "sex", "embarked", "age", "fare", "sibsp", "parch", "name"]
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic",
        version=1,
        primary_key=primary_keys,
        description="Titanic dataset")

    if BACKFILL == True:
        titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
        titanic_df.columns = [x.lower() for x in titanic_df.columns]
        titanic_df["age"] = titanic_df["age"].astype(str).astype(float)
        titanic_df["pclass"] = titanic_df["pclass"].astype(str)
        titanic_df["embarked"] = titanic_df["embarked"].fillna("?")
        titanic_df["survived"] = titanic_df["survived"].astype(bool)
        # titanic_df["synthetic"] = False
        titanic_df = titanic_df[["survived", "pclass", "sex", "embarked", "age", "fare", "sibsp", "parch", "name"]]

        titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})
    else:
        titanic_fg.insert(get_random_titanicee(), write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
