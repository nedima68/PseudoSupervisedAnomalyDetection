import os
import shutil
from pathlib import Path

fabric_codes = ['00','01','02','03','04', '05', '06']
ds_dir_names = {'def': 'defect_samples', 'no_def':'no_defect_samples', 'msk':'mask'}
current_dir = os.getcwd()        
AITEX_root = current_dir + os.sep + 'AITEX_DS'

def get_list_of_files_in_dir(directory: str, file_types: str ='*') -> list:
    return [f for f in Path(directory).glob(file_types) if f.is_file()]

def setup_dataset_dirs():
    try:
        print('current directory is : ', current_dir)
        print("creating the necessary directories...")       
        for code in fabric_codes:
            dir_name = AITEX_root + os.sep +'fabric_' + code
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            sub_dir_defect = dir_name + os.sep + ds_dir_names['def']
            sub_dir_non_defect = dir_name + os.sep + ds_dir_names['no_def']
            sub_dir_mask = dir_name + os.sep + ds_dir_names['msk']

            if not os.path.exists(sub_dir_defect):
                os.makedirs(sub_dir_defect)
            if not os.path.exists(sub_dir_non_defect):
                os.makedirs(sub_dir_non_defect)
            if not os.path.exists(sub_dir_mask):
                os.makedirs(sub_dir_mask)
    except Exception as e:
        print("ERROR when creating directories ... ", e.__class__, "occurred.")

def prepare_data():
    try:
        print("copying images to corresponding directories ...")        
        no_def = "NODefect_images"
        defect = "Defect_images"
        mask = "Mask_images"
        defect_fabric_file_set = {}
        non_defect_fabric_file_set = {}
        mask_file_set = {}
        for code in fabric_codes:
            non_defect_fabric_file_set[code] = []

        defect_files_dir = AITEX_root + os.sep + defect
        mask_files_dir = AITEX_root + os.sep + mask
        no_def_files_dir = AITEX_root + os.sep + no_def
        no_def_dirs = os.listdir(no_def_files_dir)
        for code in fabric_codes:
            defect_fabric_file_set[code] = get_list_of_files_in_dir(defect_files_dir, '*_*_' + code + '*')
            mask_file_set[code] = get_list_of_files_in_dir(mask_files_dir, '*_*_' + code + '_*')           
            for dir in no_def_dirs:
                non_defect_fabric_file_set[code].extend(get_list_of_files_in_dir(no_def_files_dir + os.sep + dir, '*_*_' + code + '*'))

        for code, file_list in defect_fabric_file_set.items():
            dir_name = AITEX_root + os.sep +'fabric_' + code
            sub_dir_defect = dir_name + os.sep + ds_dir_names['def']
            for f in file_list:
                #shutil.copy(defect_files_dir + os.sep + f, sub_dir_defect)
                shutil.copy(f, sub_dir_defect)
        print('copied all defect images to destination directories ...')

        for code, file_list in mask_file_set.items():
            dir_name = AITEX_root + os.sep +'fabric_' + code
            sub_dir_mask = dir_name + os.sep + ds_dir_names['msk']
            for f in file_list:
                #shutil.copy(defect_files_dir + os.sep + f, sub_dir_mask)
                shutil.copy(f, sub_dir_mask)

        print('copied all mask images to destination directories ...')

        for code, file_list in non_defect_fabric_file_set.items():
            dir_name = AITEX_root + os.sep +'fabric_' + code
            sub_dir_nondef = dir_name + os.sep + ds_dir_names['no_def']
            for f in file_list:
                #shutil.copy(defect_files_dir + os.sep + f, sub_dir_defect)
                shutil.copy(f, sub_dir_nondef)
        print('copied all non defect images  to destination directories ...')

        print('Finished dataset preparation ...')
    except Exception as e:
        print("ERROR when copying original images to dataset directories...", e.__class__, "occurred.")



if __name__ == '__main__':
    setup_dataset_dirs()
    prepare_data()