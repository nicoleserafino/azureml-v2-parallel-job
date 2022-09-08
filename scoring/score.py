# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

from ctypes import wstring_at
import os
import numpy as np

import azureml
from sklearn.linear_model  import LinearRegression 
from azureml.core import Model,Workspace
from azureml.core.authentication import MsiAuthentication as msi
import pandas as pd
import mlflow
from azureml.core.authentication import AzureCliAuthentication as ACA

def init():
    print ("I am inside init method")
    #ws = Workspace.from_config()
    #azureCliAuth = ACA()

    #global ws

    #msi_auth= msi()
    #subscriptionID can be passed as env variable 
    #print("this is the msi_auth",msi_auth )
    #ws = Workspace(subscription_id="411c73a9-919b-4030-bc19-4599758da714" , resource_group="RG-DEV-EUWEST-AIResearch",workspace_name="aidevelopmentML",auth=msi_auth)
    #ws = Workspace.get(subscription_id="411c73a9-919b-4030-bc19-4599758da714" , resource_group="RG-DEV-EUWEST-AIResearch", location='euwest', name="aidevelopmentML", cloud='AzureCloud',auth=msi_auth)
    #print("after workspace")
   
    #print("found workspace: ", ws.name)
    #print(ws.name) 


    global myFile
    global filePath
    global fileName
    filePath =os.getenv('outputFilePath')
    #print("my environment v", myEnv )
    fileName =os.getenv('outputFileName')

    myFile= os.path.join(filePath,fileName)
    print("myFile is ", myFile)
    if os.path.exists(myFile):
        os.remove(myFile)
        print("the file has been deleted")
    else:
        print("its a new file")
    









def run(mini_batch):
    print(f"run method start: {__file__}, run({mini_batch})")

    

    resultList = []
    
    for client_file_path in mini_batch:
        # prepare each image
        print("this is client_file_path", client_file_path)
        client_basename = os.path.basename(client_file_path)
        # process the tenant training data file to train a model + infer predictions for the final evaluation step
        clientName=client_basename.split(".")[0]
        print("inside for loop, ClientName: ", clientName)
        print("ths is client_basename", client_basename)

        scoringDate=pd.read_csv(client_file_path)

        ourModel=mlflow.sklearn.load_model("models:/paymentPred_{}/Latest".format(clientName))           
       
        scoringDate["newModel_Label"]=ourModel.predict(scoringDate.drop(columns=["DueDate","PaidDate","RaisedDate","Due_Paid_weekDelta"]))


        resultList.append("{}:{}".format(client_basename, scoringDate))
        #resultList.append(scoringDate)
    try:
        #with open(myFile,'w+') as scoredData:

             
        resultList.to_csv("predictionTest.csv",index=False)
    except:
        print ("cannot open file to write in it")

    return resultList
    #return pd.concat(resultList)
