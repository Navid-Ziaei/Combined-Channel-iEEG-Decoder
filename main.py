from src.settings import Paths, Settings
from src.data import load_data
from src.feature_extraction import FeatureExtractor
from src.model import ModelSinglePatient

# Load settings from /configs/settings.yaml
settings = Settings()
settings.load_settings()

# Load paths from /configs/device.yaml
paths = Paths(settings)
paths.load_device_paths()

# Save the updated settings with model paths
settings.save_settings(paths.path_store_model)

# Load and preprocess data
data_all_patient, channel_names_list, labels = load_data(settings, paths)

# Initialize feature extractor
feature_extractor = FeatureExtractor(
    channel_names=channel_names_list,
    path=paths,
    settings=settings
)

# Extract features for all patients
feature_extractor.get_feature_all(data=data_all_patient)

# Create the feature matrix
feature_matrix = feature_extractor.create_feature_matrix()

# Create and train the model
model = ModelSinglePatient(
    feature_matrix=feature_matrix,
    path=paths,
    settings=settings,
    channel_names_list=channel_names_list,
    label=labels
)

# Create and train the model
model.create_model()
