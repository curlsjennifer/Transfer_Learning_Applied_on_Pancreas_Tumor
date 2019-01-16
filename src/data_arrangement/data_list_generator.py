import os
import numpy as np
import pandas as pd


# Load PC info (270 rows)
PC_info = pd.read_csv('/home/d/pancreas/raw_data/PC_info.csv')
PC_info = PC_info.iloc[:, 1:]
PC_info['Code'] = PC_info['Code'].apply(lambda x: str(x).zfill(6))
PC_info['Number'] = PC_info['Number'].apply(lambda x: str(x))
PC_info['Exam Date'] = PC_info['Exam Date'].apply(lambda x: str(x)[:-2])
PC_info['Exam Date'] = PC_info['Exam Date'].apply(lambda x: pd.to_datetime(x, format='%Y%m%d', errors='ignore'))

# Load PTNP info (163 rows)
PTNP_info = pd.read_csv('/home/d/pancreas/raw_data/PTNP_info.csv')
PTNP_header = ['Code', 'Number', 'Exam Date']
PTNP_info = PTNP_info[PTNP_header]
PTNP_info['Code'] = PTNP_info['Code'].apply(lambda x: str(x).zfill(6))
PTNP_info['Number'] = PTNP_info['Number'].apply(lambda x: str(x))
PTNP_info['Exam Date'] = PTNP_info['Exam Date'].apply(lambda x: str(x))
PTNP_info['Exam Date'] = PTNP_info['Exam Date'].apply(lambda x: pd.to_datetime(x, format='%Y%m%d', errors='ignore'))

# Load AD info (101 rows)
AD_info = pd.read_csv('/home/d/pancreas/raw_data/adrenal/adrenal_list.csv')
AD_header = ['chartnumber', 'No.', 'CT Date']
AD_info = AD_info[AD_header]
AD_info = AD_info.rename(columns = {'chartnumber': 'Code', 'No.': 'Number', 'CT Date': 'Exam Date'})
AD_info['Code'] = AD_info['Code'].apply(lambda x: str(x).zfill(6))
AD_info['Number'] = AD_info['Number'].apply(lambda x: str(x))
AD_info['Exam Date'] = AD_info['Exam Date'].apply(lambda x: str(x)[:-2])
AD_info['Exam Date'] = AD_info['Exam Date'].apply(lambda x: pd.to_datetime(x, format='%Y%m%d', errors='ignore'))

# Merge all info
all_info = pd.concat([PC_info, PTNP_info, AD_info])
g = all_info.groupby(['Code'])['Exam Date', 'Number']
group_dict = dict(list(g))

# Check if the labels are in label_data (PC, PT, NP: 111, AD: 92)
final_dict = {}
label_data_path = '/home/d/pancreas/label_data/'
AD_path = '/home/d/pancreas/box_data/'
no_file_path_key = []
for key, values in group_dict.items():
    check_exists = False
    for i in range(values.shape[0]):
        code = group_dict[key].iloc[i, :]['Code']
        number = group_dict[key].iloc[i, :]['Number']
        if number[:2] =='AD':
            file_path = AD_path + number + '/pancreas.npy'
        else:
            file_path = label_data_path + code + '/' + number + '/label.nrrd'
        
        if os.path.exists(file_path):    
            final_dict[key] = values.iloc[i, :]
            check_exists = True
            break
            
    if not check_exists:
        no_file_path_key.append(key)

final_df = pd.DataFrame.from_dict(final_dict, orient='index').reset_index(drop=True)

def data_type(x):
    if (x[:2] == 'AD') or (x[:2] == 'NP'):
        return 'normal'
    else:
        return 'tumor'

final_df['Type'] = final_df['Number'].apply(lambda x: data_type(x))

# Split normal to 0.6 train, 0.2 validation, 0.2 test
normal_final_df = final_df[final_df['Type'] == 'normal']
normal_train, normal_valid, normal_test = np.split(normal_final_df.sample(frac=1), 
                                                   [int(.6*len(normal_final_df)), int(.8*len(normal_final_df))])
normal_train['Class'] = 'train'
normal_valid['Class'] = 'validation'
normal_test['Class'] = 'test'
normal_concat = pd.concat([normal_train, normal_valid, normal_test])

# Split tumor to 0.6 train, 0.2 validation, 0.2 test
tumor_final_df = final_df[final_df['Type'] == 'tumor']
tumor_train, tumor_valid, tumor_test = np.split(tumor_final_df.sample(frac=1), 
                                                [int(.6*len(tumor_final_df)), int(.8*len(tumor_final_df))])
tumor_train['Class'] = 'train'
tumor_valid['Class'] = 'validation'
tumor_test['Class'] = 'test'
tumor_concat = pd.concat([tumor_train, tumor_valid, tumor_test])

# Final output to csv
final_split_df = pd.concat([normal_concat, tumor_concat])
final_split_df = final_split_df.sort_values(by=['Code'])
final_split_df.to_csv(os.path.join('/home/d/pancreas/raw_data/data_list.csv'), index=False)

