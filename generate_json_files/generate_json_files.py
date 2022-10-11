import argparse,os
import pandas as pd

import json
# data engineering
def prepare_json_jobs(args):
    # reading data from raw data file
    startingMonth = os.getenv("startingMonth")
    history_duration=os.getenv("history_duration")

    custKey_df = pd.read_csv(args.custKey_file)
   



    with open(os.path.join(args.job_json_folder,'MLTable'),'w') as MLTable_metadata_file:
        # start to generate MLTable metadata file
       
            
        MLTable_metadata_file.write("paths:\n")
           
        for cust_id in custKey_df['CustomerKey']:

                # tenant specific data
            
            custJobdata = {
                    "CustomerKey": cust_id,
                    "startingMonth": startingMonth,
                    "history_duration": history_duration

                    }
            
           # custJobdata_dict=json.l(custJobdata)
            
            client_metadata_file_name = "customer_" + str(cust_id) + '.json'
            with open(os.path.join(args.job_json_folder,client_metadata_file_name),'w') as tenant_metadata_file:
                json.dump(custJobdata, tenant_metadata_file,index=False)
                
                print(f"{client_metadata_file_name} generated.")
                # add file to MLTable metadata file descriptor for training step
                MLTable_metadata_file.write(f"  - file: ./{client_metadata_file_name}\n")


    # alternate MLTable metadata file content (not safe if your destination folder isn't guaranteed to be empty when this runs)
    # paths:
    #  - file: ./*.csv

# read arguments
def parse_args():
    # retrieve output location
    parser = argparse.ArgumentParser()
    parser.add_argument('--custKey_file', type=str)
    parser.add_argument('--job_json_folder', type=str)
    
    args, unknown_args = parser.parse_known_args()
    print(f"custKey_file: {args.custKey_file}")
    print(f"job_json_folder: {args.job_json_folder}")
    return args

# main
if __name__ == "__main__":
    args = parse_args()
    prepare_json_jobs(args)