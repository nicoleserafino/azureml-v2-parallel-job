
SCORING_URI="https://paymentpredendpoint.westeurope.inference.ml.azure.com/jobs"
SCORING_TOKEN=$(az account get-access-token --query accessToken -o tsv)
#SCORING_TOKEN=$(az account get-access-token --echoresource https://ml.azure.com --query accessToken -o tsv)
#SCORING_TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IjJaUXBKM1VwYmpBWVhZR2FYRUpsOGxWMFRPSSIsImtpZCI6IjJaUXBKM1VwYmpBWVhZR2FYRUpsOGxWMFRPSSJ9.eyJhdWQiOiJodHRwczovL21hbmFnZW1lbnQuYXp1cmUuY29tLyIsImlzcyI6Imh0dHBzOi8vc3RzLndpbmRvd3MubmV0L2JjY2Y2ODNlLTc5NDAtNGEyZi05MDQ0LTgwNmE2ZjYwODVjMC8iLCJpYXQiOjE2NjM3Njk0OTEsIm5iZiI6MTY2Mzc2OTQ5MSwiZXhwIjoxNjYzNzczMzkxLCJhaW8iOiJFMlpnWUhDYzUxcnhucnZmaUxzNmdhUHJYa0FQQUE9PSIsImFwcGlkIjoiMzU3Y2FkZDktYTE5Yy00OTQwLWFjZTktM2I0MzRmZDE2NjIxIiwiYXBwaWRhY3IiOiIxIiwiaWRwIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvYmNjZjY4M2UtNzk0MC00YTJmLTkwNDQtODA2YTZmNjA4NWMwLyIsImlkdHlwIjoiYXBwIiwib2lkIjoiYzgyZGQyNzYtZGFlOS00ZGU1LWFhOWEtYTAwZjk3MzdkNGVkIiwicmgiOiIwLkFTOEFQbWpQdkVCNUwwcVFSSUJxYjJDRndFWklmM2tBdXRkUHVrUGF3ZmoyTUJNdkFBQS4iLCJzdWIiOiJjODJkZDI3Ni1kYWU5LTRkZTUtYWE5YS1hMDBmOTczN2Q0ZWQiLCJ0aWQiOiJiY2NmNjgzZS03OTQwLTRhMmYtOTA0NC04MDZhNmY2MDg1YzAiLCJ1dGkiOiJqWGF0OE1BZ3BFdWdodWxUUHZZYkFBIiwidmVyIjoiMS4wIiwieG1zX3RjZHQiOjE0MjMwNjkyOTF9.sV2G4Fk5xjxMxqhJ7UYNl-ScqF1dksNtuarLIN0p4ODTijdcP6ArAiap17_0xNw61nV_XFFCWJKoSQHK4eUujOWFlA5LNyOCHtHq_TZf-Qs22bYngjYwJF2TbYRt4GMmRfC-I699r1bM9bSnehc4KHUvXjHfCuvz0xJJD1ao8SKz_beie8jlSz0GVH2cCpMpIxJC7hjMBBPLi3XfWX0ArFdRTwCkXGmHA3sddXnofxWOqZLky9OFjglm6EgBtCC7HK8GzYswSg66yRRdhX64ViuhpwCMMnejRtVrcfZm33f309PT_KdIsGrEraspEZvX1XuE7C_-YxAX1jR5P6gDTg"
response=$(curl --location --request POST $SCORING_URI \
--header "Authorization: Bearer $SCORING_TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
    	\"InputData\": {
    		\"mnistinput\": {
    			\"JobInputType\" : \"UriFolder\",
    			\"Uri\":  \"azureml://datastores/aidatalake_synapse/paths/EDWData/BatchScoring\"
				}
        }
    }
}")

echo $response