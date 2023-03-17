def com_elec(raw_car_all,elec_com_fifteen,print_patient):
    elec={}
    for i in range(len(elec_com_fifteen)):
        elec[elec_com_fifteen[i]] = []

    for i in range(len(raw_car_all)):
        for key in elec.keys():
            if key in raw_car_all[i].ch_names:
                elec[key].append(i)
    if print_patient:
        for key in elec.keys():
            print('\n  ', key, '=', elec[key])

    return  elec

