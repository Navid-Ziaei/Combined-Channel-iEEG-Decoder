import os
import pickle as pkl


def load_data(settings, paths):
    if settings.task == 'Speech_Music':
        data_all_patient = []
        channel_names_list = []
        for subject in range(settings.num_patient):
            file_path_subband_data = paths.path_processed_data + 'Audio_visual/sub{}_sub-bands_data_with_hilbert.pkl'.format(
                subject)
            file_path_channel_names = paths.path_processed_data + 'Audio_visual/sub{}_channel_names.pkl'.format(subject)
            if os.path.isfile(file_path_subband_data):
                # Load the data from the pickle file
                with open(file_path_subband_data, 'rb') as f:
                    band_data_with_hilbert = pkl.load(f)
                with open(file_path_channel_names, 'rb') as f:
                    channel_names = pkl.load(f)
            else:
                raise ValueError(f"The file '{file_path_subband_data}' does not exist. Put "
                                 f"the data in this directory or"
                                 f"run the code with load_preprocessed_data=False to generate the data")
            data_all_patient.append(band_data_with_hilbert)
            channel_names_list.append(channel_names)
        with open(paths.path_processed_data + 'Audio_visual/labels.pkl', 'rb') as f:
            labels = pkl.load(f)

    elif settings.task == 'Singing_Music':
        with open(paths.path_processed_data + 'Music_Reconstruction/data_all_patient.pkl', 'rb') as f:
            data_all_patient = pkl.load(f)
        with open(paths.path_processed_data + 'Music_Reconstruction/channel_name_list.pkl', 'rb') as f:
            channel_names_list = pkl.load(f)
        with open(paths.path_processed_data + 'Music_Reconstruction/labels.pkl', 'rb') as f:
            labels = pkl.load(f)

    else:
        data_all_patient = []
        for patient in range(settings.num_patient):
            with open(paths.path_processed_data + f'Upper_Limb_Movement/patient_{patient + 1}_reformat.pkl', 'rb') as f:
                data_all_patient.append(pkl.load(f))
            with open(paths.path_processed_data + 'Upper_Limb_Movement/ch_names.pkl', 'rb') as f:
                channel_names_list = pkl.load(f)
        with open(paths.path_processed_data + 'Upper_Limb_Movement/labels.pkl', 'rb') as f:
            labels = pkl.load(f)

    return data_all_patient, channel_names_list, labels
