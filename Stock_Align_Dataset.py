# Clean and check integrity of the data. Might not be necessary now but needed in the future.


def cleaner(df_appl, feature):
    '''
    Drop "nan"s
    
    '''
    before = len(df_appl)
    df_appl = df_appl.dropna()
    after = len(df_appl)
    
    print("...{} row(s) dropped".format(before-after))
    
    return df_appl


def align(df_appl, features, debug):
    '''
    Automatically find & cluster numerical index and categorical index.
    
    '''
    
    numerical = []
    categorical = []
    
    for feature in features:
        unique_val = len(df_appl[str(feature)].unique())
        if (unique_val < 50) and (len(df_appl) >= 300):
            categorical.append(feature)
        elif (unique_val == 1) and not debug:
            print("...Found invalid featurenamed '{}'. Removing it.".format(feature))
            continue
        else:
            numerical.append(feature)
    
    categorical_index = len(numerical)
    new_feature = numerical + categorical
    
    df_aligned = df_appl[new_feature]
    
    return df_aligned, new_feature, categorical_index

def check_statistics():
    pass
    

def clean_and_align(df_appl, features, debug=False):
    df_appl = cleaner(df_appl, features)
    df_aligned, new_features, categorical_index = align(df_appl, features, debug)
    
    print("...Done !!")
    
    return df_aligned, new_features, categorical_index


