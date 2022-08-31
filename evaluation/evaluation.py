import argparse,os
from pathlib import Path
import pandas as pd
import mltable
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics import mean_absolute_percentage_error
from azureml.core import Run, Dataset,Datastore, Workspace
import joblib
from azureml.core import Model
import mlflow
import mlflow.sklearn
#from azureml_user.parallel_run import EntryScript




def init():
    # retrieve output location
    parser = argparse.ArgumentParser()
    #parser.add_argument('--predictions_mltable', type=str)
    parser.add_argument("--evalResultFolder", type=str)
    parser.add_argument("--evaluation_data_folder", type=str)
    #parser.add_argument("--model_train_runID", type=str)
    parser.add_argument("--training_log", type=str)
    parser.add_argument("--model_output_folder", type=str)
    
    global args
    args, unknown_args = parser.parse_known_args()

    #global model_train_runID
    #model_train_runID = args.model_train_runID
    
    training_log=args.training_log
    
    #print(f"predictions_mltable: {args.predictions_mltable}")
    print(f"evalResultFolder: {args.evalResultFolder}")
    #print(f"model_train_runID: {args.model_train_runID}")
    print(f"evaluation_data_folder: {args.evaluation_data_folder}")
    print(f"training_log: {args.training_log}")
    
    

    global run_id
    #global logger
    #logger = EntryScript().logger
    
    #logger.info(f"Evaluating the performance of the newely trained model on the test set")


    """with open(os.path.join(args.model_train_runID),'r') as run_ID_file:

        run_id=run_ID_file.read()
        print(run_id)   """

    """    print("this is the argument: ", model_train_00runID)
    with open((Path(model_train_runID) / "model_train_runID"), 'r') as f:

        print("I managed to open the file")
        
        run_id= f.read()
        print(run_id)"""

    with open(os.path.join(training_log),'r') as training_logFile:

        run_id_text=training_logFile.read().splitlines()[-1]
        print("the whole text is: ", run_id_text)
        run_id=run_id_text.split(" ")[-1]
        print("our golden Rund_ID is:", run_id)  

    

def run(mini_batch):
    print("I am inside run method")

    
    client = mlflow.tracking.MlflowClient()
    #for model in client.list_registered_models():
    #   print(f"{model.name}")
    #testModel = client.get_registered_model("paymentPred_tenant_2603713")

    #global run
    #global ws
    run = Run.get_context()
    ws = run.experiment.workspace

    results = []
    #logger.info(f"evluating run({mini_batch})") 

    #run_id=run.id
   
    

    for client_file_path in mini_batch:
        client_basename = os.path.basename(client_file_path)
        # process the tenant training data file to train a model + infer predictions for the final evaluation step
        clientName=client_basename.split(".")[0]
        print("inside for loop, ClientName: ", clientName)
        #mlflow.register_model("runs:/{}/model_{}".format(run_id,clientName),"paymentPred_{}".format(clientName))
        try: 
            
            #if ML Flow works properly then you load the model using run IDs
            #print("runs:/{}/model_{}".format(run_id,clientName))
            mymodelName=f"runs:/{run_id}/model_{clientName}"
            print("mymodelName: ", mymodelName)
            newModel= mlflow.sklearn.load_model(mymodelName)


            # If MLFlow does not work, then you have to load the model using joblib. This is how you do it.
            #newModel=joblib.load(os.path.join(args.model_output_folder, "model_{}.joblib".format(clientName)))
            
            print("newModel: ", newModel)
           
        except:
            print("Model not found for client: ", clientName)
           
            print("moving to next customer")
            continue
        


        current_model=None

        try:  

            #Downloading the data from Azure.
            current_modelName= f"paymentPred_{clientName}"
            
            #current_model_aml = Model(ws, current_modelName)
            #os.makedirs("current_model", exist_ok=True)
            #current_model_aml.download("current_model",exist_OK=True)
            #current_model=mlflow.sklearn.load_model(os.path.join("current_model","paymentPred_{}:1".format(clientName)))
            
            #This is first attempt
            #current_model_aml.download("current_model/model_{}".format(clientName),exist_OK=True)
            #current_model=mlflow.sklearn.load_model("current_model/model_{}".format(clientName))


            # We should be able to load the registered model using MlFlow too **** THIS HAS NOT WORKED YET ****
            current_model=mlflow.sklearn.load_model("models:/paymentPred_{}/Latest".format(clientName))

            print("current_model exists: ", current_model)
        except:
            print("current_model does not exist")
            #print("there is no current model")
            #print("register the new model")
            
            

        if current_model:
        
            #print("run current model against the evaluation dataset")
            #print("run the new model too")
            #print("if the new model was performing better")
            #print("register new model")

            with open(client_file_path,'r') as tenant__evaluate_file:
                client_evalute_df = pd.read_csv(tenant__evaluate_file)
                print("this is the datass:" )
                print(client_evalute_df)
            
            client_evalute_df["newModel_Label"]=newModel.predict(client_evalute_df.drop(columns=["DueDate","PaidDate","RaisedDate","Due_Paid_weekDelta"]))
            
            eval_mape_newModel=MAPE(client_evalute_df["Due_Paid_weekDelta"],client_evalute_df["newModel_Label"])
            print("eval_mape_newModel: ", eval_mape_newModel)



            client_evalute_df["currentModel_Label"]=current_model.predict(client_evalute_df.drop(columns=["DueDate","PaidDate","RaisedDate","Due_Paid_weekDelta","newModel_Label"]))

            eval_mape_currentModel=MAPE(client_evalute_df["Due_Paid_weekDelta"],client_evalute_df["currentModel_Label"])
            print("eval_mape_currentModel",eval_mape_currentModel)
            # If new model is performing better, then register the new model
            if eval_mape_newModel>eval_mape_currentModel:
                print("new model is better")
                mlflow.register_model("runs:/{}/model_{}".format(run_id,clientName),"paymentPred_{}".format(clientName))
                

               
               #If ML FLOW did not work and you have to register the model after saving and loading it using joblib, then this is how you save it using MLFlow:
               #mlflow.sklearn.log_model(newModel,clientName,registered_model_name="paymentPred_{}".format(clientName))
            
            else:
                print("Existing model is performing better")
                      
        else: 
            
            print("Current model does not exists + Register the new model")
            mlflow.register_model("runs:/{}/model_{}".format(run_id,clientName),"paymentPred_{}".format(clientName))
            
            #If ML FLOW did not work and you have to register the model after saving and loading it using joblib, then this is how you save it using MLFlow:
            #mlflow.sklearn.log_model(newModel,clientName,registered_model_name="paymentPred_{}".format(clientName))
            
            
            
        







        #logger.info(f"train processing({client_basename} => {client_df}) with param_1:{param_1}, env_var_1:{env_var_1}")
        # TODO: replace this part with your model training
        #time.sleep(1) 





# main
#if __name__ == "__main__":
  #  args = parse_args()
  #  evaluation(args)

    #Fixing the issue of the plot not showing up and Evaluation step is not working.
    #Register the model if there are no models already registered.
    # if there is already a model registered, then compare:
        # find a good way of comparing performance.
    # if the model is the same, then do nothing
    # if the model is performing better, then register the new model
    #use might be able to use Model.register() to register the model (copilot says)
    # use ML Flow to register and retrieve the model.
    # We will then have separate models to deploy the models.

# evaluation
def evaluation(args):
    # open up the predictions file
    predictions_mlt = mltable.load(args.predictions_mltable)
    print(f"predictions_mlt: {predictions_mlt}")
    print(f"predictions_mlt.paths: {predictions_mlt.paths}")
    predictions_df = predictions_mlt.to_pandas_dataframe()
    print(f"predictions_df: {predictions_df}")
    prepForEvaluation(predictions_df,args)




"""# read arguments
def parse_args():
    # retrieve output location
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_mltable', type=str)
    parser.add_argument("--evalResultFolder", type=str)
    args, unknown_args = parser.parse_known_args()
    print(f"predictions_mltable: {args.predictions_mltable}")
    print(f"evalResultFolder: {args.evalResultFolder}")
    return args
"""


def prepForEvaluation(allTestResults,args):
    
    allTestResults["Label"]=allTestResults["Label"].astype(float).round(0)
    allTestResults["Due_Paid_weekDelta"]=allTestResults["Due_Paid_weekDelta"].astype(float).round(0)

    allTestResults["Pred_PaidDate"] = pd.to_datetime(allTestResults["DueDate"]) +pd.to_timedelta(allTestResults["Label"],unit='W')
    
    #Create Year Week for both actual paid date and predicted paid date so that we can then do sum up payments.
    allTestResults["Year_week_Paid"]=pd.to_datetime(allTestResults["PaidDate"]).dt.strftime("%Y%W")
    allTestResults["Year_week_PredPaid"]=pd.to_datetime(allTestResults["Pred_PaidDate"]).dt.strftime("%Y%W")
    print("Data Types: \n ", allTestResults.dtypes)
    allTestResults["OriginalInvoiceAmount"]=allTestResults["OriginalInvoiceAmount"].astype(float)

    sumPayment_Paid=allTestResults.groupby("Year_week_Paid")["OriginalInvoiceAmount"].agg(['sum']).rename(columns={'sum':'PAmount_total_PaidDay'})
    sumPayment_Paid.reset_index(inplace=True)
    sumPayment_PredPaid=allTestResults.groupby("Year_week_PredPaid")["OriginalInvoiceAmount"].agg(['sum']).rename(columns={'sum':'PAmount_total_PredDay'})
    sumPayment_PredPaid.reset_index(inplace=True)
    print(sumPayment_Paid.head(4))
    print(sumPayment_PredPaid.head(4))



    #Creating a merged DF to make plotting simpler.
    finalComparison = sumPayment_Paid.merge(sumPayment_PredPaid, how="outer", left_on="Year_week_Paid", right_on="Year_week_PredPaid",suffixes=('_left', '_right'), sort=True, indicator=True)

    finalComparison["PAmount_total_PaidDay"]=finalComparison["PAmount_total_PaidDay"].astype(float)
    finalComparison["PAmount_total_PredDay"]=finalComparison["PAmount_total_PredDay"].astype(float)

    #After mergin get only rows/dates for which there was an actual payment (not predicted only), so exclude rows/dates where there is no actual payment.
    theEvaluationSet=finalComparison[finalComparison["_merge"]!="right_only"]

    print(len(theEvaluationSet["PAmount_total_PredDay"]))
    print(len(theEvaluationSet["PAmount_total_PaidDay"]))

    print("Predicted Payment: ", list(theEvaluationSet["PAmount_total_PredDay"]))
    print("Actual Payment: ", list(theEvaluationSet["PAmount_total_PaidDay"]))

    #Inevitably after excluding rows/dates where there is no actual payment (i.e. get rows for which there is some actual payments), there will be some rows/dates where there is no predicted payment and hence will be NaN.
    theEvaluationSet["PAmount_total_PredDay"]=theEvaluationSet["PAmount_total_PredDay"].fillna(0)

    print("Predicted Payment: ", list(theEvaluationSet["PAmount_total_PredDay"]))
    print("Actual Payment: ", list(theEvaluationSet["PAmount_total_PaidDay"]))


    score = mean_absolute_percentage_error(theEvaluationSet["PAmount_total_PaidDay"], theEvaluationSet["PAmount_total_PredDay"])
    score = score.round(2)
    print(f"MAPE score is : {score}")






    finalComparison['Year_week_Paid'] = np.where(finalComparison['Year_week_Paid'].isnull(), finalComparison['Year_week_PredPaid'], finalComparison['Year_week_Paid'])
    finalComparison['Year_week_PredPaid'] = np.where(finalComparison['Year_week_PredPaid'].isnull(), finalComparison['Year_week_Paid'], finalComparison['Year_week_PredPaid'])

   # actualPaymentLength=len(sumPayment_Paid["PAmount_total_PaidDay"])
    

    with open(os.path.join(args.evalResultFolder,"evalResult"),'w') as Eval_file:
        plot = finalComparison.plot(x="Year_week_Paid", y=["PAmount_total_PaidDay","PAmount_total_PredDay"], kind="bar",figsize=(14,14))
        fig = plot.get_figure()
        #fig.savefig(" {}.jpg ".format(Eval_file))
                #Eval_file.write("evalResult.png")
        fig.savefig(os.path.join(Eval_file,".png"))
        Eval_file.write("MAPE score is : {}".format(score))




def MAPE(Y_actual,Y_Predicted):

    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape


