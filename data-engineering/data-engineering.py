import argparse,os
import pandas as pd
from sklearn.model_selection import train_test_split
# data engineering
def prepare_training_data(args):
    # reading data from raw data file
    raw_data_df = pd.read_csv(args.raw_data_file, nrows=10000)
    # TODO: data engineering here for what can be done accross all tenants in the source data...
    raw_data_df.rename(columns = {'InvoiceCount':'TotalInvoiceCount', 'Value':'OriginalInvoiceAmount'}, inplace = True)
    raw_data_df=raw_data_df.loc[raw_data_df["OriginalInvoiceAmount"]>0]
    raw_data_df["Due_Paid_weekDelta"] = raw_data_df["WofY_Paid"]-raw_data_df["WofY_Due"]

    #Drop columns that are not needed
    raw_data_df = raw_data_df.drop(columns=["DofM_Paid","MofY_Paid","MofY_Paid","WofY_Paid","DofWInMonth_Paid",
    "CurrentInstanceDofWInMonth_Paid", "isFirstInstanceDayOfWeekInMonth_Paid","isLastInstanceDayOfWeekInMonth_Paid",
    "PaymentType", "DofW_Paid", "TopPredictedPayDate", "Predicted","InvoiceBandValue","TotalInvoices",
    "Prediction_Accuracy", "Raised_Paid_Variance",  "Due_Paid_Variance","lastDueM_Paid_Variance","StatementTransactionType",
    "Due_TopPredictedPayDate_Variance","PercentageOfPaymentsUnknown","DofWInMonth_Due","DofWInMonth_Raised"] )

 
    #SORT DATA BASED ON PAID DATE
    raw_data_df.sort_values(by=["PaidDate"],inplace=True)
    raw_data_df= raw_data_df.drop_duplicates(subset=["ClientID","InvoiceID"],keep="last")
    
    #Drop Date to make a simple LR model possible
    #raw_data_df=raw_data_df.drop(columns=["DueDate","PaidDate","RaisedDate"])



    with open(os.path.join(args.training_data_folder,'MLTable'),'w') as MLTable_metadata_file:
        # start to generate MLTable metadata file
        with open(os.path.join(args.evaluation_data_folder,'MLTable'),'w') as MLTable_Evaluation_metadata_file:
            
            MLTable_metadata_file.write("paths:\n")
            MLTable_Evaluation_metadata_file.write("paths:\n")


            
            for client_id in raw_data_df['ClientID'].unique()[:2]:
                # tenant specific data
                customer_data_df = raw_data_df[raw_data_df['ClientID'] == client_id]


                # TODO: potential extra data engineering at the tenant level

                # extract a evaluation dataset for final evaluation
                #finalEvalDataset=customer_data_df.iloc[0:-]
                train, finalEvalDataset = train_test_split(customer_data_df, test_size=0.2,shuffle=False)

            



                # now save tenant training data to file into training data folder
                client_metadata_file_name = "tenant_" + str(client_id) + '.csv'
                with open(os.path.join(args.training_data_folder,client_metadata_file_name),'w') as tenant_metadata_file:
                    train.to_csv(tenant_metadata_file,index=False)
                
                print(f"{client_metadata_file_name} generated.")
                # add file to MLTable metadata file descriptor for training step
                MLTable_metadata_file.write(f"  - file: ./{client_metadata_file_name}\n")


                # now save tenant evaluation data to file into evaluation data folder
                with open(os.path.join(args.evaluation_data_folder,client_metadata_file_name),'w') as tenant_evaluation_metadata_file:
                    finalEvalDataset.to_csv(tenant_evaluation_metadata_file,index=False)


                print(f"Evaluation: {client_metadata_file_name} generated.")
                MLTable_Evaluation_metadata_file.write(f"  - file: ./{client_metadata_file_name}\n")

    # alternate MLTable metadata file content (not safe if your destination folder isn't guaranteed to be empty when this runs)
    # paths:
    #  - file: ./*.csv

# read arguments
def parse_args():
    # retrieve output location
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_file', type=str)
    parser.add_argument('--training_data_folder', type=str)
    parser.add_argument('--evaluation_data_folder', type=str)
    args, unknown_args = parser.parse_known_args()
    print(f"raw_data_file: {args.raw_data_file}")
    print(f"training_data_folder: {args.training_data_folder}")
    return args

# main
if __name__ == "__main__":
    args = parse_args()
    prepare_training_data(args)