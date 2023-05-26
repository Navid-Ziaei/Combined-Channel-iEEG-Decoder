from collections import Counter


def electrode_histohram(channel_names_list, print_analyze):
    all_electrodes_names = []
    for i in range(len(channel_names_list)):
        all_electrodes_names.extend(channel_names_list[i])
    h = Counter(all_electrodes_names)

    elec_more_one = []
    elec_more_ten = []
    elec_more_fifteen = []
    elec_more_tweny = []
    for key in h.keys():
        if h[key] > 1:
            elec_more_one.append(key)
        if h[key] > 10:
            elec_more_ten.append(key)
        if h[key] > 15:
            elec_more_fifteen.append(key)
        if h[key] > 20:
            elec_more_tweny.append(key)
    if print_analyze:
        print('number of unique electrodes is =', len(h), '\n max number of electrode repetition=', 23)
        print('number of shared electrode in more than one patient = ', len(elec_more_one))
        print('number of shared electrode in more than ten patient = ', len(elec_more_ten))
        print('number of shared electrode in more than fifteen patient = ', len(elec_more_fifteen))
        print('number of shared electrode in more than twenty patient = ', len(elec_more_tweny))
        print('\n\n', elec_more_fifteen, 'Electrodes shared between more the 15 patients')
        print('', elec_more_tweny, 'Electrodes shared between more the 20 patients')
    return h, elec_more_fifteen
