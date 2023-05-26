from collections import Counter

def hist_elec(raw_car_all,print_analyze):
    d = []
    for i in range(len(raw_car_all)):
        elec_list = raw_car_all[i].ch_names
        d = [*d, *elec_list]
    h = Counter(d)

    elec_more_one = []
    elec_more_ten = []
    elec_more_fifteen = []
    elec_more_tweny = []
    for key in h.keys():
        if (h[key] > 1):
            elec_more_one.append(key)
        if (h[key] > 10):
            elec_more_ten.append(key)
        if (h[key] > 15):
            elec_more_fifteen.append(key)
        if (h[key] > 20):
            elec_more_tweny.append(key)
    if print_analyze:
        print('number of total electrode is =', len(h), '\nmax number of electrod is same=', 23)
        print('number of electrod that are same in more than one patient = ', len(elec_more_one))
        print('number of electrod that are same in more than ten patient = ', len(elec_more_ten))
        print('number of electrod that are same in more than fifteen patient = ', len(elec_more_fifteen))
        print('number of electrod that are same in more than tweny patient = ', len(elec_more_tweny))
        print('\n\n', elec_more_fifteen, 'elec_more_fifteen\n\n')
        print(elec_more_tweny, 'elec_more_tweny')
    return h,elec_more_fifteen


