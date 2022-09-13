import os
import tempfile
import logging
from azureml.core.model import Model
import joblib
import pandas as pd
from azureml.core import Run,Dataset
import os
from datetime import date

def init():
    global ws, datastore, output_folder_prep
    run = Run.get_context()
    ws = run.experiment.workspace

    run = Run.get_context()
    ws = run.experiment.workspace
    datastore_name = os.getenv("DATASTORE")


    datastore = ws.datastores[datastore_name]
    output_folder_prep = os.getenv("OUTPUT_FOLDER")


def run(mini_batch):
    print(f"run method start: {__file__}, run({mini_batch})")
    result_list =[]
    for batch in mini_batch:
        basename = os.path.basename(batch)
        Inputs = pd.read_csv(batch)

        timestamp_column= 'WeekStarting'
        Inputs[timestamp_column]=pd.to_datetime(Inputs[timestamp_column])
        timeseries_id_columns= [ 'Store', 'Brand']
        data = Inputs \
            .set_index(timestamp_column) \
            .sort_index(ascending=True)

        model_name = basename[5:-4]
        print("model name from path ", model_name)
        model = Model(ws,model_name)
        model.download(exist_ok=True)
        # with open("model.pkl", 'rb') as file:  
        #     model = pickle.load(file)
        model = joblib.load(model_name)
        print("model loaded")
        # predictions = model.forecast(data)
        # data["prediction"] = predictions
        
        # try:
        #     data.drop(['Unnamed: 0'], axis=1,inplace=True)
        # except:
        #     pass
        # file_name = batch.split("/")[-1]
        # folder_name = str(date.today())

        # output_folder = os.path.join(output_folder_prep, folder_name)

        target = (datastore,output_folder_prep)

        # write to Azure storage
        with tempfile.TemporaryDirectory() as tmpdir:
            data.to_csv(f'{tmpdir}/{model_name}.csv', index=False)
            Dataset.File.upload_directory(tmpdir,target,overwrite=True)
        result_list.append({"file_name":model_name, "file_size":data.shape[0]})

    return result_list