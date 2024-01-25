import os.path


def tumorbank_filename_to_patient_id(filename: str) -> str:
    filename = os.path.split(os.path.splitext(filename)[0])[1]
    return int(filename.split('_')[0])


def tumorbank_filename_to_stain_name(filename: str) -> str:
    filename = os.path.split(os.path.splitext(filename)[0])[1]
    return filename.split('_')[3].split('-')[0]


def otta_filename_to_patient_id(filename: str) -> str:
    filename = os.path.split(os.path.splitext(filename)[0])[1]
    return filename.split('_')[1]


def otta_filename_to_stain_name(filename: str) -> str:
    filename = os.path.split(os.path.splitext(filename)[0])[1]
    return filename.split('_')[0]


def jhu_filename_to_patient_id(filename: str) -> str:
    filename = os.path.split(os.path.splitext(filename)[0])[1]
    return filename.split('_')[1]


def jhu_filename_to_stain_name(filename: str) -> str:
    filename = os.path.split(os.path.splitext(filename)[0])[1]
    return filename.split('_')[0]


def filename_to_patient_id(filename: str, dataset_name: str):
    # todo: convert name to function name
    dataset_name = dataset_name.lower()
    if dataset_name == 'tumorbank':
        return tumorbank_filename_to_patient_id(filename)
    if dataset_name.startswith('otta'):
        return otta_filename_to_patient_id(filename)
    if dataset_name == 'jhu':
        return jhu_filename_to_patient_id(filename)
    raise RuntimeError("invalid dataset type")


def filename_to_stain_name(filename: str, dataset_name: str):
    # todo: convert name to function name
    dataset_name = dataset_name.lower()
    if dataset_name == 'tumorbank':
        return tumorbank_filename_to_stain_name(filename)
    if dataset_name.startswith('otta'):
        return otta_filename_to_stain_name(filename)
    if dataset_name == 'jhu':
        return jhu_filename_to_stain_name(filename)
    raise RuntimeError("invalid dataset type")
