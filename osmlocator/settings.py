def getDefaultSetting():
    ss = {}
    ss['supportable_markers'] = 'o,o_,s,s_,D,D_,^,^_,v,v_,6,7,+,*' 
    ss['space_setting_factor'] = 60
    ss['is_auto_search_space'] = True
    ss['impact_of_recognized_marks'] = 0.3
    ss['gamma_markov'] = 1.5
    ss['gamma_stop'] = 1.5
    ss['alpha'] = 1.1
    ss['beta'] = 1
    ss['verbose'] = False
    return ss
