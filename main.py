from src.settings import Paths, Settings
from src.data import load_data
from src.feature_extraction import FeatureExtractor
from src.model import ModelSinglePatient

# Loading Settings from /configs/settings.yaml
settings = Settings()
settings.load_settings()

# Loading Paths from /configs/device.yaml
paths = Paths(settings)
paths.load_device_paths()


# Load and preprocess data
data_all_patient, channel_names_list, labels = load_data(settings, paths)


feature_ex = FeatureExtractor(channel_names=channel_names_list,
                              path=paths,
                              settings=settings)

feature_ex.get_feature_all(data=data_all_patient)

feature_all_matrix = feature_ex.create_feature_matrix()

# type_classification : 'log_reg' or 'SVM' or 'Naive_bayes'
# type_balancing :  'over_sampling' or 'under_sampling'  or 'over&down_sampling' or 'weighted_losfunc'
# notice that for classification Naive_bayes don't use 'weighted_losfunc' way for balancing
model = ModelSinglePatient(feature_matrix=feature_all_matrix,
                           path=paths,
                           settings=settings,
                           channel_names_list=channel_names_list)
model.create_model()


