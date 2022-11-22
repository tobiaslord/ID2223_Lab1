import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def get_image_url(survived):
    if survived:
        return "https://m.media-amazon.com/images/I/71M6k7ZQNcL._RI_.jpg"
    else:
        return "https://thumbs.dreamstime.com/b/allvarlig-sten-med-skallen-34707626.jpg"

def get_survived_str(survived):
    if survived:
        return "Survived!"
    else:
        return "Died!"

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    print("LOGGED in?")
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("titanic", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")

    titanic_fg = fs.get_feature_group(name="titanic", version=1)
    df = titanic_fg.read()
    last_row = df.iloc[-1].to_frame().T
    x = last_row[["pclass", "sex", "embarked", "age", "fare", "sibsp", "parch"]]
    y = last_row[["survived"]]
    print(last_row)
    y_val = y.iloc[0]["survived"]

    feature_view = fs.get_feature_view(name="titanic", version=1)
    batch_data = feature_view.get_batch_data()


    y_pred = model.predict(x)
    print("Predicted:", get_survived_str(y_pred))

    predicted_url = get_image_url(y_pred)
    img = Image.open(requests.get(predicted_url, stream=True).raw)
    img.save("./latest_prediction.png")

    dataset_api = project.get_dataset_api()
    dataset_api.upload("./latest_prediction.png", "Resources/images", overwrite=True)

    titanic_fg = fs.get_feature_group(name="titanic", version=1)
    df = titanic_fg.read()
    print("Actual: " + get_survived_str(y_val))
    label_url = get_image_url(y_val)
    img = Image.open(requests.get(label_url, stream=True).raw)
    img.save("./latest_input.png")
    dataset_api.upload("./latest_input.png", "Resources/images", overwrite=True)

    n_rows = len(df.index)
    print("n_rows", n_rows)
    if n_rows > 892:
        added_rows = n_rows - 892
        print("added rows", added_rows)
        monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                    version=1,
                                                    primary_key=["datetime"],
                                                    description="Titanic Prediction/Outcome Monitoring"
                                                    )

        now = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")

        # Add our prediction to the history, as the history_df won't have it -
        # the insertion was done asynchronously, so it will take ~1 min to land on App
        data = {
            'prediction': [y_pred],
            'label': [y_val],
            'datetime': [now],
        }
        monitor_df = pd.DataFrame(data)

        monitor_df["prediction"] = monitor_df["prediction"].astype(bool)
        monitor_df["label"] = monitor_df["label"].astype(bool)
        monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})

        history_df = monitor_fg.read()
        if len(history_df.index) - added_rows == 0:
            #No new rows
            return
        history_df = pd.concat([history_df, monitor_df])
        history_df.sort_values(by="datetime", ascending=False)


        df_recent = history_df.tail(5)
        dfi.export(df_recent, './df_recent_titanic.png', table_conversion = 'matplotlib')
        dataset_api = project.get_dataset_api()
        dataset_api.upload("./df_recent_titanic.png", "Resources/images", overwrite=True)

        predictions = history_df[['prediction']]
        labels = history_df[['label']]

        # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
        print("Number of different titanic predictions to date: " + str(predictions.value_counts().count()))
        if predictions.value_counts().count() == 2:
            results = confusion_matrix(labels, predictions)

            df_cm = pd.DataFrame(results, ['True Died', 'True Survived'],
                                ['Pred Died', 'Pred Survived'])
            cm = sns.heatmap(df_cm, annot=True)
            fig = cm.get_figure()
            fig.savefig("./confusion_matrix_titanic.png")
            dataset_api.upload("./confusion_matrix_titanic.png", "Resources/images", overwrite=True)
        else:
            print("You need 2 predictions to create the confusion matrix.")
            print("Run the batch inference pipeline more times until you get 2 different survival predictions")


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
