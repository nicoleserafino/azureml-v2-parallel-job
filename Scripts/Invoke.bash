
SCORING_URI="https://paymentpredendpoint.westeurope.inference.ml.azure.com/jobs"
TOKEN=$(az account get-access-token --query accessToken -o tsv)
SCORING_TOKEN=$(az account get-access-token --resource https://ml.azure.com --query accessToken -o tsv)
response=$(curl --location --request POST $SCORING_URI \
--header "Authorization: Bearer $SCORING_TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
    	\"InputData\": {
    		\"mnistinput\": {
    			\"JobInputType\" : \"UriFolder\",
    			\"Uri\":  \"https://aidatalake1.blob.core.windows.net/aidatalake1/EDWData/parallel-pipeline/amlv2_pj_evaluation_data\"
    		}
        }
    }
}")

echo $response