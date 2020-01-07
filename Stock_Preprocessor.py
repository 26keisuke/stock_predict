import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import math
import time

from sklearn.preprocessing import MinMaxScaler

def to_sequence(df, seq_length, categorical):
    df_seq = []
    df_label = []
    
    for i in range(seq_length, len(df)):
        df_seq.append(df[i-seq_length:i])
        df_label.append(df[i])
        
    if categorical:
        return df_seq, None
    
    return df_seq, df_label

def permute_and_reshape(df_seq, df_label, categorical_index, non_label_index, test_enabled=False):
    '''
    df_seq & df_label: list of sequence data ("Open", "Close", "High", ...)
    *Caution: permutation must remain the same across the same row("Open", "Close", "High", ...)
    
    '''
    length = len(df_label[0]) # number of datasets
    perm = np.random.permutation(length)
    
    # loop through features in datasets
    for idx, feature in enumerate(df_seq):
        
        categorical = check_categorical(idx, categorical_index)
        non_label = check_non_label(idx, non_label_index)
        
        df_seq[idx] = np.array(df_seq[idx])
        
        if non_label:
            pass
        elif categorical:
            pass
        else:
            df_label[idx] = np.array(df_label[idx])
           
        if test_enabled == False:
            df_seq[idx] = df_seq[idx][perm]
            if non_label:
                pass
            elif categorical:
                pass
            else:
                df_label[idx] = df_label[idx][perm]
        else:
            continue

    return df_seq, df_label

def window_scaler(df_seq, df_label, seq_length, scaler, categorical):
    '''
    Use scaler within seqence length
    
    '''
    
    if categorical:
        pass
    else:
        for idx, seq in enumerate(df_seq):
            temp = np.concatenate((df_seq[idx], df_label[idx].reshape(-1, 1)), axis=0)
            scaled = scaler.fit_transform(temp)
            df_seq[idx] = scaled[0:seq_length]
            df_label[idx] = scaled[-1]  
    
    return df_seq, df_label

def check_categorical(idx, categorical_index):
    '''
    Check if idx is categorical value. eg) Time, Hour, Month...
    
    '''
    if categorical_index == 0:
        pass
    elif idx >= categorical_index:
        return True 
    
    return False

def check_non_label(idx, non_label_index):
    '''
    Check if idx is categorical value. eg) Time, Hour, Month...
    
    '''
    if non_label_index == -1:
        pass
    elif (type(non_label_index) == int) and (idx == non_label_index):
        return True 
    elif (type(non_label_index) == list) and (idx in non_label_index):
        return True
    
    return False
    

def transform(df_added, data_feature, feature_range, seq_length, val, test, categorical_index, non_label_index):
    
    assert test % seq_length == 0, "Must be common divider !"
    assert val % seq_length == 0, "Must be common divider !"
    
    df_transformed = [] # close, open, high, low
    
    meta = data_feature # Must be returned !
    
    scaler = MinMaxScaler(feature_range=feature_range)
    
    train_df, train_seq, train_label= [], [], []
    val_df, val_seq, val_label= [], [], []
    test_df, test_seq, test_label= [], [], []
    
    train_interval = (len(df_added) - (val + test)) // seq_length * seq_length # 41940
    val_interval = val + train_interval
    
    # loop through each data_feature eg) "Open", "Close", "Time Stamp"...
    for idx, feature in enumerate(data_feature):
        
        print("...Transforming feature_{}".format(idx))
        
        categorical = check_categorical(idx, categorical_index)

        df_transformed.append(df_added[feature].values.reshape(-1, 1).astype(float))
        
        train_df.append(df_transformed[idx][:train_interval])
        val_df.append(df_transformed[idx][train_interval:val_interval])
        test_df.append(df_transformed[idx][val_interval:val_interval+test])
        
        train_seq_temp, train_label_temp = to_sequence(train_df[idx], seq_length, categorical)
        train_seq_window, train_label_window = window_scaler(train_seq_temp,
                                                                     train_label_temp, seq_length, scaler, categorical)
        train_seq.append(train_seq_window)
        train_label.append(train_label_window)
        
        val_seq_temp, val_label_temp = to_sequence(val_df[idx], seq_length, categorical)
        val_seq_window, val_label_window = window_scaler(val_seq_temp,
                                                                     val_label_temp, seq_length, scaler, categorical)
        val_seq.append(val_seq_window)
        val_label.append(val_label_window)
        
        test_seq_temp, test_label_temp = to_sequence(test_df[idx], seq_length, categorical)
        test_seq_window, test_label_window = window_scaler(test_seq_temp,
                                                                       test_label_temp, seq_length, scaler, categorical)
        test_seq.append(test_seq_window)
        test_label.append(test_label_window)  
        
    train_seq, train_label = permute_and_reshape(train_seq, train_label, categorical_index, non_label_index)
    val_seq, val_label = permute_and_reshape(val_seq, val_label, categorical_index, non_label_index)
    test_seq, test_label = permute_and_reshape(test_seq, test_label, categorical_index, non_label_index, True)
    
    return train_seq, train_label, val_seq, val_label, test_seq, test_label, test_df, meta, scaler

def input_concat(df, df_label, categorical_index, non_label_index):
    '''
    Concatenate across axis=2 (input_size) for multi dimensional
   
    '''
    subject = []
    subject_label = []
    
    for idx, feature in enumerate(df):
        subject.append(feature)
        
        categorical = check_categorical(idx, categorical_index)
        non_label = check_non_label(idx, non_label_index)
        
        if (non_label == True):
            pass
        elif (categorical == True):
            pass
        else:
            subject_label.append(df_label[idx])

    concat_tuple = tuple(subject)
    concat_label_tuple = tuple(subject_label)
    
    df_concatenated = np.concatenate(concat_tuple, axis=2)
    df_label_concatenated = np.concatenate(concat_label_tuple, axis=1)
    
    return df_concatenated, df_label_concatenated
    
# Change this to generator function in the future to resume from where its left off
def process(df_aligned, feature, feature_range, seq_length, val, test, categorical_index, non_label_index=-1): 
    
    start = time.time()

    assert len(df_aligned) >= 300, "Dataset must be bigger than 300 (or equal)!"
    assert len(df_aligned) > (val + test), "Dataset is too small to make val & test sets! Consider using smaller numbers."
    assert len(df_aligned) > seq_length, "Dataset is too small to make sequences! Consider using smaller numbers."
    assert seq_length + 1 < val, "Validation set must be larger than seq_length + 1"
    assert seq_length + 1 < test, "Test set must be larger than seq_length + 1"
    assert seq_length + 1 < len(df_aligned) - (val + test), "Training set is too small" 
    
    if categorical_index != 0:
        print("...Categorical Index Enabled")
        print()
    
    print("...Transforming Datasets")
    train_seq, train_label, val_seq, val_label, test_seq, test_label, test_df, meta, scaler =\
                                        transform(df_aligned, feature, feature_range, seq_length, val, test, categorical_index, non_label_index)
    
    print()
    print("...Concatenating Datasets")
    train, train_label = input_concat(train_seq, train_label, categorical_index, non_label_index)
    val, val_label = input_concat(val_seq, val_label, categorical_index, non_label_index)
    test, test_label = input_concat(test_seq, test_label, categorical_index, non_label_index)
    
    time_took = time.time() - start
    
    print()
    print("...Done ! Took {:.1f}s to complete.".format(time_took))
    
    return train, train_label, val, val_label, test, test_label, test_df, meta, scaler