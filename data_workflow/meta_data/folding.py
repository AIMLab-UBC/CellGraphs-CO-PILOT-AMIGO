import sklearn


def create_folds(study_ids, k):
    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)
    folds = []
    for i, (train_idx, test_idx) in enumerate(sklearn.model_selection.KFold(n_splits=k, shuffle=False).split(study_ids)):
        print(f'fold {i}, train_idx: {repr(study_ids[train_idx])}, test_idx: {repr(study_ids[test_idx])}')
        folds.append((study_ids[train_idx], study_ids[test_idx]))
    return folds


def get_folded_study_ids(study_ids, k, dataset_name):
    dataset_name = dataset_name.lower()
    folds = create_folds(study_ids, k)
    return folds


def get_heldout_study_ids(dataset_name):
    dataset_name = dataset_name.lower()
    return None


def get_heldout_training_study_ids(dataset_name):
    dataset_name = dataset_name.lower()
    return None