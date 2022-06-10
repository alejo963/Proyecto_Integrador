import json
from tkinter.tix import InputOnly
import boto3
import pandas as pd
import numpy as np
import zipfile
from os import listdir
from time import time

SCALER_ENDPOINT_NAME = "sklearn-scaler-local-ep"
ESTIMATOR_ENDPOINT_NAME = "sklearn-logit-local-ep"


def extract_datasets(path_to_zip_file):
    """
    Extract both datasets from a zip file and returns a single clean dataframe
    """
    # Extract CSV files and save in DataFrames
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall('/tmp/')

    csvFiles = [
        f'/tmp/{file}' for file in listdir('/tmp') if file.endswith('.csv')]
    if len(csvFiles) != 2:
        return None, True
    df1 = pd.read_csv(csvFiles[0])
    df2 = pd.read_csv(csvFiles[1])

    # Merge dataframes
    df = pd.merge(df1, df2, how='inner', on='TransactionID')
    cols_to_preserve = ['TransactionID', 'id_01', 'id_02', 'id_05', 'id_06', 'id_11', 'id_12', 'id_15', 'id_17', 'id_19', 'id_20', 'id_28', 'id_29', 'id_31', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195',
                        'V196', 'V197', 'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321']

    # Select columns and drop nulls
    df = df[cols_to_preserve]
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, False


def getNumericVariables(df):
    """
    Retrieve numeric variables from dataframe
    """
    # Get numeric variables
    numeric_df = df.select_dtypes('number').drop(
        ['TransactionID', 'TransactionDT'], axis=1)
    num_cols = numeric_df.columns

    return numeric_df


def getCategoricVariables(df):
    cat_df = df.select_dtypes(exclude='number').copy()
    cat_df.id_31 = cat_df.id_31.map(lambda x: x.lower())
    cat_df.id_31 = cat_df.id_31.map(lambda x: 'chrome' if 'chrome' in x else x)
    cat_df.id_31 = cat_df.id_31.map(
        lambda x: 'firefox' if 'firefox' in x else x)
    cat_df.id_31 = cat_df.id_31.map(lambda x: 'safari' if 'safari' in x else x)
    cat_df.id_31 = cat_df.id_31.map(lambda x: 'edge' if 'edge' in x else x)
    cat_df.id_31 = cat_df.id_31.map(lambda x: 'opera' if 'opera' in x else x)
    cat_df.id_31 = cat_df.id_31.map(
        lambda x: 'samsung' if 'samsung' in x else x)
    cat_df.id_31 = cat_df.id_31.map(
        lambda x: 'android' if 'android' in x else x)
    cat_df.id_31 = cat_df.id_31.map(
        lambda x: 'google' if 'google ' in x else x)
    cat_df.id_31 = cat_df.id_31.map(lambda x: 'ie' if 'ie ' in x else x)
    cat_df.id_31 = cat_df.id_31.map(
        lambda x: 'other' if x not in 'chrome firefox safari edge opera samsung android google ie'.split() else x)

    # Recat P_emaildomain and R_emaildomain
    def transform(x):
        try:
            return x[0: x.index('.')]
        except:
            return x

    cat_df.P_emaildomain = cat_df.P_emaildomain.map(transform)
    choosen_cats_P = ['gmail', 'hotmail', 'anonymous', 'yahoo', 'aol', 'outlook', 'comcast',
                      'live', 'msn', 'icloud', 'verizon', 'sbcglobal']
    df.P_emaildomain = df.P_emaildomain.map(
        lambda x: x if x in choosen_cats_P else 'other')

    df.R_emaildomain = df.R_emaildomain.map(transform)
    choosen_cats_R = ['gmail', 'hotmail', 'anonymous', 'yahoo', 'aol', 'outlook', 'comcast',
                      'live', 'icloud', 'msn', 'sbcglobal', 'verizon']
    df.R_emaildomain = df.R_emaildomain.map(
        lambda x: x if x in choosen_cats_R else 'other')

    final_cat_df = pd.get_dummies(cat_df, drop_first=False)
    all_cats = ['id_12_Found', 'id_12_NotFound', 'id_15_Found', 'id_15_New',
                'id_15_Unknown', 'id_28_Found', 'id_28_New', 'id_29_Found',
                'id_29_NotFound', 'id_31_android', 'id_31_chrome', 'id_31_edge',
                'id_31_firefox', 'id_31_google', 'id_31_ie', 'id_31_opera',
                'id_31_other', 'id_31_safari', 'id_31_samsung', 'id_35_F', 'id_35_T',
                'id_36_F', 'id_36_T', 'id_37_F', 'id_37_T', 'id_38_F', 'id_38_T',
                'DeviceType_desktop', 'DeviceType_mobile', 'ProductCD_C', 'ProductCD_H',
                'ProductCD_R', 'card4_american express', 'card4_discover',
                'card4_mastercard', 'card4_visa', 'card6_charge card', 'card6_credit',
                'card6_debit', 'P_emaildomain_anonymous', 'P_emaildomain_aol',
                'P_emaildomain_comcast', 'P_emaildomain_gmail', 'P_emaildomain_hotmail',
                'P_emaildomain_icloud', 'P_emaildomain_live', 'P_emaildomain_msn',
                'P_emaildomain_other', 'P_emaildomain_outlook',
                'P_emaildomain_sbcglobal', 'P_emaildomain_verizon',
                'P_emaildomain_yahoo', 'R_emaildomain_anonymous', 'R_emaildomain_aol',
                'R_emaildomain_comcast', 'R_emaildomain_gmail', 'R_emaildomain_hotmail',
                'R_emaildomain_icloud', 'R_emaildomain_live', 'R_emaildomain_msn',
                'R_emaildomain_other', 'R_emaildomain_outlook',
                'R_emaildomain_sbcglobal', 'R_emaildomain_verizon',
                'R_emaildomain_yahoo']

    for col in final_cat_df.columns:
        if col not in all_cats:
            final_cat_df.drop(col, axis=1, inplace=True)
    for col in final_cat_df:
        final_cat_df[col] = final_cat_df.groupby(col)[col].transform(
            lambda x: x/np.sqrt(x.count()/len(df)))
    for cat in all_cats:
        if cat not in final_cat_df:
            final_cat_df[cat] = 0
    final_cat_df = final_cat_df[all_cats]
    return final_cat_df


def dropCorrelatedVariables(df):
    """
    Drop all correlated variables found in previous analysis
    """
    refined_df = df.copy()
    uncorr_var = ['id_01', 'id_02', 'id_05', 'id_06', 'id_19', 'id_20', 'TransactionAmt',
                  'card1', 'card2', 'card3', 'card5', 'C14', 'V170', 'V203', 'V229',
                  'V245', 'V263', 'V280', 'V282', 'V308', 'id_12_Found', 'id_15_New',
                  'id_29_NotFound', 'id_31_android', 'id_31_edge', 'id_31_firefox',
                  'id_31_ie', 'id_31_opera', 'id_31_other', 'id_31_safari',
                  'id_31_samsung', 'id_36_T', 'id_37_T', 'id_38_F', 'DeviceType_desktop',
                  'ProductCD_R', 'card4_american express', 'card4_discover',
                  'card4_mastercard', 'card6_credit', 'P_emaildomain_anonymous',
                  'P_emaildomain_aol', 'P_emaildomain_comcast', 'P_emaildomain_hotmail',
                  'P_emaildomain_icloud', 'P_emaildomain_msn', 'P_emaildomain_other',
                  'P_emaildomain_verizon', 'P_emaildomain_yahoo',
                  'R_emaildomain_anonymous', 'R_emaildomain_aol', 'R_emaildomain_comcast',
                  'R_emaildomain_icloud', 'R_emaildomain_live', 'R_emaildomain_msn',
                  'R_emaildomain_other', 'R_emaildomain_outlook',
                  'R_emaildomain_sbcglobal', 'R_emaildomain_verizon',
                  'R_emaildomain_yahoo']

    for col in df.columns:
        if col not in uncorr_var:
            refined_df.drop(col, axis=1, inplace=True)

    return refined_df


def lambda_handler(event, context):
    unique_id = int(time())
    sagemakerRuntime = boto3.client('runtime.sagemaker')
    s3 = boto3.resource('s3')

    # Download zipfile with csv
    key = event['Records'][0]['s3']['object']['key']
    bucket = s3.Bucket('detechito-datalake')
    path_to_zip_file = '/tmp/archive.zip'
    bucket.download_file(key, path_to_zip_file)

    raw_df, error = extract_datasets(path_to_zip_file)

    if error:
        return {
            'statusCode': 400,
            'message': 'Zip file must contain just two CSV files'
        }

    num_df = getNumericVariables(raw_df)
    scaler_payload = {"Input": json.dumps(num_df.to_numpy().tolist())}
    # Scaling through SageMaker
    scaler_response = sagemakerRuntime.invoke_endpoint(EndpointName=SCALER_ENDPOINT_NAME,
                                                       ContentType='application/json',
                                                       Body=scaler_payload)

    scaler_result = json.loads(
        scaler_response['Body'].read().decode())['Output']
    scaled_num_df = pd.DataFrame(
        np.array(scaler_result), columns=num_df.columns)

    cat_df = getCategoricVariables(raw_df)

    full_df = pd.concat([scaled_num_df, cat_df], axis=1)

    refined_df = dropCorrelatedVariables(full_df)

    estimator_payload = {"Input": json.dumps(refined_df.to_numpy().toList())}
    # Get predictions
    estimator_response = sagemakerRuntime.invoke_endpoint(EndpointName=ESTIMATOR_ENDPOINT_NAME,
                                                          ContentType='application/json',
                                                          Body=estimator_payload)

    estimator_result = json.loads(
        estimator_response['Body'].read().decode())['Output']

    predictions = pd.concat(
        [raw_df['TransactionID'], pd.DataFrame(estimator_result)], axis=1)

    predictions.to_csv(f'/tmp/{unique_id}predictions.csv')

    return {
        'statusCode': 200,
        'body': json.dumps('Predictions saved in bucket')
    }
