import argparse,os,time
from pathlib import Path
import pandas as pd
from azureml_user.parallel_run import EntryScript
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from azureml.core import Model
import mlflow
import mlflow.sklearn
from azureml.core import Run, Dataset,Datastore, Workspace
import joblib

# init() called once per process
def init():
  
    # retrieve parameters from the entry script
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_1", type=str)
    parser.add_argument("--predictions_data_folder", type=str)
    parser.add_argument("--model_train_runID", type=str)
    parser.add_argument("--model_output_folder", type=str)
    parser.add_argument("--silver_data_folder", type=str)
    args, unknown = parser.parse_known_args()
    global param_1
    global predictions_data_folder
    global model_train_runID
    param_1 = args.param_1
    predictions_data_folder = args.predictions_data_folder
    model_train_runID=args.model_train_runID
    
    global model_output_folder
    model_output_folder = args.model_output_folder

    silver_data_folder = args.silver_data_folder


    # retrieve environment variables
    global env_var_1
    env_var_1 = os.environ['env_var_1']

    # AML logger
    global logger
    logger = EntryScript().logger
    logger.info(f"train init(), param_1:{param_1}, env_var_1:{env_var_1}")

    run = Run.get_context()
    ws = run.experiment.workspace
    global run_id
    run_id = run.id
    #with open(os.path.join(model_train_runID),'w') as modelIDFile:
    # log to global results object
    #    modelIDFile.write(run_id)
    
    print("run_id is {}".format(run_id))

    with open(os.path.join(model_train_runID,"model_train_runID"),'w') as model_train_runIDFile:
        print("At training, run id is: ", run_id)
        model_train_runIDFile.write(run_id)

    """    with open((Path(model_train_runID) / "model_train_runID.txt"), 'w') as f:
               
        f.write(run_id)
        print("At training, run id is: ", run_id)
    """
   
# run() called as many times as needed to process all files in the parallel job input
def run(mini_batch):
    
    results = []
    logger.info(f"train run({mini_batch})")
    for client_file_path in mini_batch:
       
       
    """client_basename = os.path.basename(client_file_path)
        # process the tenant training data file to train a model + infer predictions for the final evaluation step
        with open(client_file_path,'r') as tenant_file:
            client_df = pd.read_csv(tenant_file)
        logger.info(f"train processing({client_basename} => {client_df}) with param_1:{param_1}, env_var_1:{env_var_1}")
        # TODO: replace this part with your model training
        #time.sleep(1)"""

    client_basename = os.path.basename(client_file_path)
     # process the tenant training data file to train a model + infer predictions for the final evaluation step
    with open(client_file_path,'r') as tenant_file:  

        customerPath_in_silverData="READ Json linebyline create the path"
        fullPath_DataLake=os.path.join(silver_data_folder,customerPath_in_silverData)


    with open(fullPath_DataLake,'r') as tenant_file:
        client_df = pd.read_csv(tenant_file)



        """        y=client_df["Due_Paid_weekDelta"]
        x = client_df.drop(columns="Due_Paid_weekDelta")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

        model = LinearRegression().fit(x_train.drop(columns=["DueDate","PaidDate","RaisedDate"]), y_train)"""
        



        X, y, sample_weights = prepare_data(client_df) 
        split_ratio = 0.25
        (X_train, y_train, sample_weights_train), (X_valid, y_valid, sample_weights_valid) = split_dataset(X, y, sample_weights, split_ratio, should_stratify=False)
        model = train_model(X_train.drop(columns=["DueDate","PaidDate","RaisedDate"]), y_train, sample_weights_train)





        mlflow.sklearn.log_model(model, "model_{}".format(client_basename.split(".")[0]))

        
        #joblib.dump(model, os.path.join(model_output_folder,"model_{}.joblib".format(client_basename.split(".")[0])))

        #myRunID= mlflow.active_run().info.run_id
        #print("this is myRunID: ", myRunID)
        
        #test_result = model.predict(x_test.drop(columns=["DueDate","PaidDate","RaisedDate"]))
        
        test_result = model.predict(X_valid.drop(columns=["DueDate","PaidDate","RaisedDate"]))

        # TODO: write the output of your models predictions for further performance evaluation of your strategy
        outputToReturn=X_valid.assign(y_test=y_valid)
        
        outputToReturn["Label"]=test_result
        with open(os.path.join(predictions_data_folder,client_basename),'w') as prediction_file:
            outputToReturn.to_csv(prediction_file,index=False)


        
        results.append(f"{client_basename} processed")
        
    results.append(f"run_Id is {run_id}")
    return results

 
 
def setup_instrumentation():
    import logging
    import sys

    from azureml.core import Run
    from azureml.telemetry import INSTRUMENTATION_KEY, get_telemetry_log_handler
    from azureml.telemetry._telemetry_formatter import ExceptionFormatter

    logger = logging.getLogger("azureml.training.tabular")

    try:
        logger.setLevel(logging.INFO)

        # Add logging to STDOUT
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)

        # Add telemetry logging with formatter to strip identifying info
        telemetry_handler = get_telemetry_log_handler(
            instrumentation_key=INSTRUMENTATION_KEY, component_name="azureml.training.tabular"
        )
        telemetry_handler.setFormatter(ExceptionFormatter())
        logger.addHandler(telemetry_handler)

        # Attach run IDs to logging info for correlation if running inside AzureML
        try:
            run = Run.get_context()
            parent_run = run.parent
            return logging.LoggerAdapter(logger, extra={
                "properties": {
                    "codegen_run_id": run.id,
                    "parent_run_id": parent_run.id
                }
            })
        except Exception:
            pass
    except Exception:
        pass

    return logger


def get_mapper_b86fd6(column_names):

    from azureml.training.tabular.featurization.categorical.cat_imputer import CatImputer
    from azureml.training.tabular.featurization.datetime.datetime_transformer import DateTimeFeaturesTransformer
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': CatImputer,
                'copy': True,
            },
            {
                'class': StringCastTransformer,
            },
            {
                'class': DateTimeFeaturesTransformer,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    

def generate_data_transformation_config():
    from sklearn.pipeline import FeatureUnion
    
    column_group_1 = ['RaisedDate', 'DueDate']
    
    mapper = get_mapper_b86fd6(column_group_1)
    return mapper
    
    
def generate_preprocessor_config():
    from sklearn.preprocessing import StandardScaler
    
    preproc = StandardScaler(
        copy=True,
        with_mean=True,
        with_std=True
    )
    
    return preproc
    
    
def generate_algorithm_config():

    #from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import LinearRegression
    """  algorithm = ExtraTreesRegressor(
        bootstrap=False,
        ccp_alpha=0.0,
        criterion='mse',
        max_depth=None,
        max_features=0.5,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        min_samples_leaf=0.004196633747563344,
        min_samples_split=0.000753222139758624,
        min_weight_fraction_leaf=0.0,
        n_estimators=25,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )"""


    algorithm = LinearRegression()


    
    return algorithm



#   ('featurization', generate_data_transformation_config()),
def build_model_pipeline():
    from sklearn.pipeline import Pipeline
    
    logger.info("Running build_model_pipeline")
    pipeline = Pipeline(
        steps=[
            ('preproc', generate_preprocessor_config()),
            ('model', generate_algorithm_config()),
        ]
    )
    
    return pipeline

def split_dataset(X, y, weights, split_ratio, should_stratify):
    from sklearn.model_selection import train_test_split

    random_state = 42
    if should_stratify:
        stratify = y
    else:
        stratify = None

    if weights is not None:
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
        weights_train, weights_test = None, None

    return (X_train, y_train, weights_train), (X_test, y_test, weights_test)

def prepare_data(dataframe):
    from azureml.training.tabular.preprocessing import data_cleaning
    
    logger.info("Running prepare_data")
    label_column_name = 'Due_Paid_weekDelta'
    
    # extract the features, target and sample weight arrays
    y = dataframe[label_column_name].values
    X = dataframe.drop([label_column_name], axis=1)
    sample_weights = None
    X, y, sample_weights = data_cleaning._remove_nan_rows_in_X_y(X, y, sample_weights,
     is_timeseries=False, target_column=label_column_name)
    
    return X, y, sample_weights


def train_model(X, y, sample_weights=None, transformer=None):
    logger.info("Running train_model")
    model_pipeline = build_model_pipeline()
    
    model = model_pipeline.fit(X, y)
    return model
