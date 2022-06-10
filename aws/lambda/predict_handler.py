import json
import boto3
import pandas as pd
import numpy as np
import zipfile
from os import listdir
sagemakerRuntime = boto3.client('runtime.sagemaker')
s3 = boto3.resource('s3')


def extract_datasets(path_to_zip_file):
    """
    Extract both datasets from a zip file and saves a single clean dataframe
    ready for scaling in .csv format
    """
    # Extract CSV files and save in DataFrames
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall('/tmp/')
    csvFiles = [f'/tmp/{file}' for file in listdir('/tmp') if file.endswith('.csv')]
    df1 = pd.read_csv(csvFiles[0])
    df2 = pd.read_csv(csvFiles[1])
    
    # Merge dataframes
    df = pd.merge(df1, df2, how='inner', on='TransactionID')
    cols_to_preserve = ['TransactionID','id_01','id_02','id_05','id_06','id_11','id_12','id_15','id_17','id_19','id_20','id_28','id_29','id_31','id_35','id_36','id_37','id_38','DeviceType','TransactionDT','TransactionAmt','ProductCD','card1','card2','card3','card4','card5','card6','P_emaildomain','R_emaildomain','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','D1','V95','V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111','V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126','V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V167','V168','V169','V170','V171','V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186','V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201','V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216','V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231','V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246','V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261','V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276','V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291','V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306','V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321']
    
    # Select columns and drop nulls
    df = df[cols_to_preserve]
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Get numeric variables
    numeric_df = df.select_dtypes('number').drop(['TransactionID', 'TransactionDT'], axis=1)
    num_cols = numeric_df.columns
    numeric_df.to_csv('/tmp/df.csv')
    

def lambda_handler(event, context):
    print(event)
    # Download zipfile with csv
    key = event['Records'][0]['s3']['object']['key']
    bucket = s3.Bucket('detechito-datalake')
    path_to_zip_file = '/tmp/archive.zip'
    bucket.download_file(key, path_to_zip_file)
    
    extract_datasets(path_to_zip_file)
    
    response = sagemakerRuntime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                                ContentType='text/csv',
                                                Body=payload)
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
