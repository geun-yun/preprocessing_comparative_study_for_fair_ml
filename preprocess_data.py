import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import kagglehub
import os


def icd9_chapter(code):
    if pd.isna(code):
        return "unknown"

    s = str(code).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return "unknown"

    # E / V codes
    if s[0] in {"E", "e"}:
        return "external"        # external causes of injury
    if s[0] in {"V", "v"}:
        return "supplemental"    # supplemental classification

    try:
        x = float(s)
    except ValueError:
        return "unknown"

    major = int(np.floor(x))

    if   1   <= major <= 139: return "infectious"
    elif 140 <= major <= 239: return "neoplasms"
    elif 240 <= major <= 279: return "endocrine"
    elif 280 <= major <= 289: return "blood"
    elif 290 <= major <= 319: return "mental"
    elif 320 <= major <= 389: return "nervous"
    elif 390 <= major <= 459: return "circulatory"
    elif 460 <= major <= 519: return "respiratory"
    elif 520 <= major <= 579: return "digestive"
    elif 580 <= major <= 629: return "genitourinary"
    elif 630 <= major <= 679: return "pregnancy"
    elif 680 <= major <= 709: return "skin"
    elif 710 <= major <= 739: return "musculoskeletal"
    elif 740 <= major <= 759: return "congenital"
    elif 760 <= major <= 779: return "perinatal"
    elif 780 <= major <= 799: return "symptoms"
    elif 800 <= major <= 999: return "injury"
    else:
        return "unknown"


def chapter_to_supergroup(ch):
    # keep E/V/unknown separate
    if ch in {"external", "supplemental", "unknown", "infectious", "neoplasms"}:
        return ch

    if ch in {"endocrine", "blood"}:
        return "metabolic"
    if ch in {"mental", "nervous"}:
        return "neuro_psych"
    if ch in {"circulatory", "respiratory"}:
        return "cardio_resp"
    if ch in {"digestive", "genitourinary"}:
        return "gi_gu"
    if ch in {"pregnancy", "perinatal"}:
        return "pregnancy_perinatal"
    if ch in {"musculoskeletal", "skin", "congenital"}:
        return "msk_skin_congenital"
    if ch in {"symptoms", "injury"}:
        return "symptoms_injury"

    return "unknown"

def encode_diag_with_ohe(df, diag_cols=("diag_1", "diag_2", "diag_3"),drop_original=True):
    df = df.copy()
    sg_cols = []

    for col in diag_cols:
        sg = f"{col}_sg"
        df[sg] = df[col].apply(icd9_chapter).apply(chapter_to_supergroup)
        sg_cols.append(sg)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=int)
    enc = ohe.fit_transform(df[sg_cols])
    names = ohe.get_feature_names_out(sg_cols)

    enc_df = pd.DataFrame(enc, columns=names, index=df.index)
    df = pd.concat([df, enc_df], axis=1)

    if drop_original:
        df = df.drop(columns=list(diag_cols) + sg_cols)

    return df

def legit_spec(dataset, norm=False):
    if dataset == "aids":
        return ("symptom", None)
    elif dataset == "tumor":
        return ("IDH1", None)
    elif dataset == "visits":
        return ("Physical_Health", lambda x: (x < 4).astype(int))
    elif dataset == "income":
        return ("education.num", lambda x: (x > 12).astype(int)) # checked
    elif dataset == "diabetes":
        return ("time_in_hospital", lambda x: (x > 4).astype(int))
    elif dataset == 'stroke':
        return ("hypertension", None)
    elif dataset == "liver":
        return("LiverFunctionTest", lambda x: (x > 60).astype(int)) # checked
    elif dataset == "heart":
        return("currentSmoker", None) # checked
    elif dataset == 'cad':
        return("heart_failure", lambda x: (x > 1).astype(int))
    elif dataset == 'loan':
        return("loan_percent_income", lambda x: (x < 0).astype(int))
    elif dataset == 'default':
        return("X6", lambda x: (x > 0).astype(int)) # checked
    elif dataset == 'coupon':
        return("RestaurantLessThan20", lambda x: (x > 2).astype(int)) # checked
    elif dataset == 'compas':
        return('RecSupervisionLevel', lambda x: (x > 2).astype(int)) # checked
    else:
        raise ValueError("No legit variable specified.")

def print_distribution(df, feature):
    counts = df[feature].value_counts(dropna=False)
    percent = counts / counts.sum() * 100
    dist = pd.DataFrame({
        'count': counts,
        'percentage (%)': percent.round(2)
    })
    print(dist)

# AIDs (23, 2139), 'symptom': https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175
# Doctor visits (14, 714), 'Physical_Health' < 4: https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha)
# Brain tumor (23, 839), 'IDH1': https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset
# Income (44, 30162) 'education.num' > 9: https://www.kaggle.com/datasets/uciml/adult-census-income
# Diabetes (85, 99493) 'time_in_hospital' > 4: https://www.kaggle.com/datasets/brandao/diabetes/data?select=diabetic_data.csv
def get_data(dataset):
    if dataset == 'aids':
        id = 890
    elif dataset == 'visits':
        id = 936
    elif dataset == 'tumor':
        id = 759
    elif dataset == 'income':
        df = pd.read_csv('datasets/adult.csv', na_values='?')
        df = df.dropna()
    elif dataset == 'diabetes':
        df = pd.read_csv('datasets/diabetic_data.csv', na_values='?')
    elif dataset == 'stroke':
        df = pd.read_csv('datasets/brain_stroke.csv')
    elif dataset == 'liver':
        df = pd.read_csv('datasets/Liver_disease_data.csv')
    elif dataset == 'heart':
        path = kagglehub.dataset_download("dileep070/heart-disease-prediction-using-logistic-regression")
        os.listdir(path)
        df = pd.read_csv(f'{path}/framingham.csv')
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
    elif dataset == "cad":
        df = pd.read_csv('datasets/DataClean-fullage.csv')
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
    elif dataset == "loan":
        df = pd.read_csv('datasets/loan_data.csv')
        df.dropna(inplace=True)
    elif dataset == "default":
        id = 350
    elif dataset == "coupon":
        id = 603
    elif dataset == "compas":
        df = pd.read_csv('datasets/compas-scores-raw.csv')
        df.dropna(inplace=True) 

    if dataset not in ['income', 'diabetes', 'stroke', 'liver', 'heart', 'cad', 'loan', 'compas']:
        data = fetch_ucirepo(id=id)
        X = data.data.features
        y = data.data.targets
        df = pd.concat([X,y], axis=1)

    print(f'\nDataset {dataset}, dimension: {df.shape}:')
    # print(df.isnull().values.any())
    # print(data.metadata)
    # print(data.varicables)
    # print(df.head())

    # Standardise Gender (0: female, 1: male) and race (0: non-white, 1: white)
    # Encode categorical features via Ordinal or One-hot
    if dataset == 'aids':
        df = df.rename(columns={'gender': 'Gender', 'race': 'Race', 'cid': 'Target'})
        df['Race'] = df['Race'].replace({0: 1, 1: 0})
    elif dataset == 'visits':
        df = df.rename(columns={'Gender': 'Gender', 'Number_of_Doctors_Visited': 'Target'})
        df['Gender'] = df['Gender'].replace({2: 0})
        df['Race'] = df['Race'].map(lambda x: 1 if x == 1 else 0)
        df['Target'] = df['Target'].map(lambda x: 0 if x == 1 else 1)
        df = df.drop(df[df['Physical_Health'] == -1].index)
    elif dataset == 'tumor':
        df = df.rename(columns={'Gender': 'Gender', 'Grade': 'Target'})
        df['Gender'] = df['Gender'].replace({0: 1, 1: 0})
        df['Race'] = df['Race'].map(lambda x: 1 if x == 'white' else 0)
    elif dataset == 'income':
        df = df.rename(columns={'sex': 'Gender', 'income': 'Target'})
        df['Gender'] = (df['Gender'] == 'Male').astype(int)
        df['Target'] = (df['Target'] == '>50K').astype(int)
        df = df.drop(columns=['education', 'fnlwgt'])
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col])
    elif dataset == 'diabetes':
        df = encode_diag_with_ohe(df, diag_cols=("diag_1", "diag_2", "diag_3"), drop_original=True)
        df = df.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult', 'nateglinide','chlorpropamide', 
                              'acetohexamide', 'tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 
                              'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                              'metformin-rosiglitazone', 'metformin-pioglitazone', 'diabetesMed'])
        df = df.dropna()
        df = df.rename(columns={'gender': 'Gender', 'race': 'Race', 'readmitted': 'Target'})
        df = df[df['Gender'] != 'Other']
        df['Gender'] = (df['Gender'] == 'Male').astype(int)
        df['Race'] = (df['Race'] == 'Caucasian').astype(int)
        df['Target'] = (df['Target'] != 'NO').astype(int)
        age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
        age_enc = OrdinalEncoder(categories=[age_order]) 
        df['age'] = age_enc.fit_transform(df[['age']]).astype(int)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        encoder = OneHotEncoder(sparse_output=False, dtype=int)
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, encoded_df], axis=1)
    elif dataset == 'stroke':
        df = df.rename(columns={'gender': 'Gender', 'stroke': 'Target'})
        df['Gender'] = (df['Gender'] == 'Male').astype(int)

        df['ever_married'] = (df['ever_married'] == 'Yes').astype(int)
        df['Residence_type'] = (df['Residence_type'] == 'Urban').astype(int)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        encoder = OneHotEncoder(sparse_output=False, dtype=int)
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, encoded_df], axis=1)

        scaler = StandardScaler()

        norm_cols = ['age', 'avg_glucose_level', 'bmi']
        df[norm_cols] = scaler.fit_transform(df[norm_cols])
    elif dataset == "liver":
        df = df.rename(columns={'Diagnosis': 'Target'})
        df['Gender'] = df['Gender'].replace({0: 1, 1: 0})
    elif dataset == "heart":
        df = df.rename(columns={'male': 'Gender', 'TenYearCHD': 'Target'})
    elif dataset == "cad":
        df = df.rename(columns={'gender': 'Gender', 'cad': 'Target'})
        df['Gender'] = (df['Gender'] == 'M').astype(int)
        df['type'] = (df['type'] == 'E').astype(int)

        le = LabelEncoder()
        df['outcome'] = le.fit_transform(df['outcome'])

        order = ['low', 'normal', 'high']
        enc = OrdinalEncoder(categories=[order]) 
        df['group_plate'] = enc.fit_transform(df[['group_plate']]).astype(int)
        df['group_leuk'] = enc.fit_transform(df[['group_leuk']]).astype(int)

        order = ['c_belowNormal', 'd_normal', 'b_mildHF', 'a_severeHF']
        enc = OrdinalEncoder(categories=[order]) 
        df['group_ejectf'] = enc.fit_transform(df[['group_ejectf']]).astype(int)

        order = ['0-30', '31-45', '46-60', '61-75', '76-150']
        enc = OrdinalEncoder(categories=[order]) 
        df['group_age'] = enc.fit_transform(df[['group_age']]).astype(int)
    elif dataset == "loan":
        df = df.rename(columns={'person_gender': 'Gender', 'loan_status': 'Target'})
        df['Gender'] = (df['Gender'] == 'male').astype(int)
        order = ['High School', 'Associate', 'Bachelor', 'Master','Doctorate']
        enc = OrdinalEncoder(categories=[order]) 
        df['person_education'] = enc.fit_transform(df[['person_education']]).astype(int)

        le = LabelEncoder()
        df['person_home_ownership'] = le.fit_transform(df['person_home_ownership'])

        df['loan_intent'] = le.fit_transform(df['loan_intent'])
        df['previous_loan_defaults_on_file'] = (df['previous_loan_defaults_on_file'] == 'Yes').astype(int)
    elif dataset == "default":
        df = df.rename(columns={'X2': 'Gender', 'Y': 'Target'})
        df['Gender'] = (df['Gender'] == 1).astype(int)
    elif dataset == "coupon":
        df.drop(columns=['car', 'toCoupon_GEQ5min'], axis=1, inplace=True)
        df.dropna(subset=df.columns, inplace=True)
        cat_cols = ['destination', 'passenger', 'weather', 'coupon', 'maritalStatus', 'occupation']
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col])
        df = df.rename(columns={'gender': 'Gender', "Y": "Target"})
        df['Gender'] = (df['Gender'] == 'Male').astype(int)

        time_map = {"7AM": 7, "10AM": 10, "2PM": 14, "6PM": 18, "10PM": 22}
        df["time"] = df["time"].map(time_map)

        expiration_map = {"1d": 24, "2h": 2}
        df['expiration'] = df['expiration'].map(expiration_map)

        age_map = {'50plus': 51, 'below21': 16}
        df['age'] = df['age'].replace(age_map)
        df['age'] = (df['age']).astype(int)

        educatiaon_order = ['Some High School', 'High School Graduate', 'Some college - no degree', 'Associates degree', 'Bachelors degree', 'Graduate degree (Masters or Doctorate)']
        enc = OrdinalEncoder(categories=[educatiaon_order]) 
        df['education'] = enc.fit_transform(df[['education']]).astype(int)

        freq_order = ['never', 'less1', '1~3', '4~8', 'gt8']
        enc = OrdinalEncoder(categories=[freq_order]) 
        df['Bar'] = enc.fit_transform(df[['Bar']]).astype(int)
        df['CoffeeHouse'] = enc.fit_transform(df[['CoffeeHouse']]).astype(int)
        df['CarryAway'] = enc.fit_transform(df[['CarryAway']]).astype(int)
        df['RestaurantLessThan20'] = enc.fit_transform(df[['RestaurantLessThan20']]).astype(int)
        df['Restaurant20To50'] = enc.fit_transform(df[['Restaurant20To50']]).astype(int)

        income_map = {"$25000 - $37499": 31250, "$12500 - $24999": 18750, "$100000 or More": 100000, 
                      "$37500 - $49999": 43750, "$50000 - $62499": 56250, "Less than $12500": 12500, 
                      "$62500 - $74999": 68750, "$87500 - $99999": 93750, "$75000 - $87499": 81250}
        df['income'] = df['income'].map(income_map)

        # for col in df.columns:
        #     print_distribution(df, col)
        # print(df.head())
    elif dataset == "compas":
        df.drop(columns=['Person_ID', 'AssessmentID', 'Case_ID', 'DisplayText', 'ScoreText', 'IsCompleted', 'IsDeleted', 'AssessmentReason', 'ScaleSet', 'RecSupervisionLevelText'], axis=1, inplace=True)
        df = df.rename(columns={'Sex_Code_Text': 'Gender', "DecileScore": "Target"})
        df['Gender'] = (df['Gender'] == 'Male').astype(int)
        df['Target'] = (df['Target'] > 5).astype(int)
        cat_cols = ['AssessmentType', 'MaritalStatus', 'CustodyStatus', 'LegalStatus', 'Language', 'Ethnic_Code_Text', 'MiddleName', 'FirstName', 'LastName', 'Agency_Text']
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col])
        df["Screening_Date_Year"] = pd.to_datetime(df["Screening_Date"], errors="coerce").dt.year
        df["Screening_Date_Month"] = pd.to_datetime(df["Screening_Date"], errors="coerce").dt.month
        df["Screening_Date_Day"] = pd.to_datetime(df["Screening_Date"], errors="coerce").dt.day
        df["DateOfBirth_Year"] = pd.to_datetime(df["DateOfBirth"], errors="coerce").dt.year
        df["DateOfBirth_Month"] = pd.to_datetime(df["DateOfBirth"], errors="coerce").dt.month
        df["DateOfBirth_Day"] = pd.to_datetime(df["DateOfBirth"], errors="coerce").dt.day
        df.drop(columns=['Screening_Date', 'DateOfBirth'], axis=1, inplace=True)
        # for col in df.columns:
        #     print_distribution(df, col)
        # print(df.head())
    # print(df.dtypes)
    # print(df.head())
    print_distribution(df, 'Gender')
    # print_distribution(df, 'Race')
    print_distribution(df, 'Target')
    print(f'The dimension is now {df.shape}')
    # print(df.iloc[0])
    return df

# df = get_data('compas')
# df = get_data('heart').copy()
# selector = VarianceThreshold(threshold=0.1)
# df = selector.fit_transform(df)
# df = pd.DataFrame(df)
# print(df)
# print(df.shape)
# df = get_data('income')
# df.to_csv('preprocessed_income.csv', index=False)
# if np.any(np.isnan(df)) or np.any(np.isinf(df)):
#     print("Data contains NaNs or Infs")