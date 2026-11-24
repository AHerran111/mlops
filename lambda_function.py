import os
import boto3
import json
import pandas as pd
import numpy as np
from io import StringIO

s3_client = boto3.client("s3", region_name=boto3.Session().region_name)
bucket_name = "sagemaker-us-east-2-205952664736"
cp_dict_name = "Data/Clean/cp_dict.csv"
shf_name = "Data/Clean/shf.csv"
CORS = {"Access-Control-Allow-Origin": "*"}

runtime = boto3.client('runtime.sagemaker')
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

def lambda_handler(event, context):
    try:
        if event.get("httpMethod") == "GET" or event.get("requestContext", {}).get("http", {}).get("method") == "GET":
            q = event.get("queryStringParameters") or {}
            request = {
                "Municipality": q["Municipality"],
                "CP": int(q["CP"]),
                "Year": int(q["Year"]),
                "Quarter": int(q["Quarter"]),
                "Terrain": float(q["Terrain"]),
                "Type": q["Type"],
            }

        else:
            
            data = json.loads(event['body'])
            request = data.get('request')
            
        if not request:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Request is required'})
            }

        payload = {
            "inputs": [request]
        }

        object = s3_client.get_object(Bucket=bucket_name, Key=cp_dict_name)
        body = object['Body']
        csv_string = body.read().decode('utf-8')
        cp_dict = pd.read_csv(StringIO(csv_string))

        object = s3_client.get_object(Bucket=bucket_name, Key=shf_name)
        body = object['Body']
        csv_string = body.read().decode('utf-8')
        shf_df = pd.read_csv(StringIO(csv_string))
       
        
        df = pd.DataFrame(payload['inputs'],columns = ["Municipality","CP","Year","Quarter","Terrain","Type"])
        ['Price', 'Terrain', 'Quarter', 'Year', 'shf_index', 'lat', 'lon',
    'closest_gamma', 'Type_Casa', 'Type_Departamento', 'Type_casa',
    'Municipality_GUADALAJARA', 'Municipality_TLAJOMULCO',
    'Municipality_TLAQUEPAQUE', 'Municipality_ZAPOPAN']
        
        df["Terrain"] = np.log(df["Terrain"])
        df['Municipality_GUADALAJARA'] = 0 
        df['Municipality_TLAJOMULCO'] = 0 
        df['Municipality_TLAQUEPAQUE'] = 0 
        df['Municipality_ZAPOPAN'] = 0 
        df['Type_Departamento'] = 0 
        df['Type_Casa'] = 0 

        
        if df.at[0,"Type"] == 'CASA':
            df['Type_Casa'] = 1
        else :
            df['Type_Departamento'] = 1

        
        if df.at[0,"Municipality"] == 'GUADALAJARA':
            df['Municipality_GUADALAJARA'] = 1
        elif df.at[0,"Municipality"] == 'TLAJOMULCO':
            df['Municipality_TLAJOMULCO'] = 1
        elif df.at[0,"Municipality"] == 'TLAQUEPAQUE':
            df['Municipality_TLAQUEPAQUE'] = 1
        else :
            df['Municipality_ZAPOPAN'] = 1

        
        cp_feats = cp_dict[["closest_gamma", "CP",'lat','lon']].drop_duplicates(subset="CP")

        df = df.merge(cp_feats, on="CP", how="left")
        df.drop(columns=['CP'],inplace=True)
        df = df.merge(
        shf_df[["Year", "Municipality", "Quarter", "shf_index"]],
        on=["Quarter", "Year", "Municipality"],
        how="left",
        )

      

        df.drop(columns = ["Municipality",'Type'],inplace=True)
                # Same column order as training (x_df without "Price")
        FEATURES = ['Terrain', 'Quarter', 'Year', 'shf_index', 'lat', 'lon',
                    'closest_gamma', 'Type_Casa', 'Type_Departamento', 'Type_casa',
                    'Municipality_GUADALAJARA', 'Municipality_TLAJOMULCO',
                    'Municipality_TLAQUEPAQUE', 'Municipality_ZAPOPAN']

        df['Type_casa'] = 0   # this column existed in training, so it must be sent too


        X = df[FEATURES].astype(float).values.tolist()   # [[...14 numbers...]]

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps(X)
        )

        response_body = json.loads(response['Body'].read().decode())
        print(response_body)
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'prediction': response_body})
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e),'request' :data})
        }