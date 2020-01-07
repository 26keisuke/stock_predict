import torch
import numpy as np

import matplotlib.pyplot as plt

def select_model(num):
    # dataset must be resized
    # input_size must be resized
    pass 

def fast_plot(df, df_predicted):
    plt.figure(figsize=(21, 12))
    plt.plot(df, color="r")
    if type(df_predicted) != type(None):
        plt.plot(df_predicted, color="b")
    plt.xlabel("Time")
    plt.ylabel("Price_Normalized")
    plt.title("Stock_Price")
    plt.show()

def find_outlier(df, column):
    idx_list = list()
    prev = df[column].iloc[0]
    for idx, row in enumerate(df[column]):
        if abs(float(row) - float(prev)) > 1:
            idx_list.append(idx)
        prev = row
    return idx_list

def remove_non_label(df, non_label_index):
    df = np.concatenate((df[:, :, :non_label_index], df[:, :, non_label_index+1:]), axis=2)
    return df
    
def shrink_df(df, shrink_to):
    df = df[:, :, :shrink_to]
    return df

def result_compare(model, df, df_label, idx_list, features,  bt_size, non_label_index=-1, shrink_to=-1, remove_non_label_index=False, verbose=True):
    
    if remove_non_label_index == True:
        df = remove_non_label(df, non_label_index)
        
    if shrink_to != -1:
        df = shrink_df(df, shrink_to) 
        
    def compare_and_print(df, df_label, idx, features, non_label_index, bt_size, verbose):
            subject = torch.from_numpy(df[idx:idx+bt_size, :, :])
            predict = (subject).float()
            predicted = model(predict) # (64, 4)
            target = df_label[idx:idx+bt_size, :]
            previous = df[idx:idx+bt_size, -1, :non_label_index]

            pred_res = (predicted - torch.from_numpy(target).float()).abs().sum()
            base_res = (torch.from_numpy(target).float() - torch.from_numpy(previous).float()).abs().sum()
            
            if verbose == False:
                return (base_res - pred_res) / base_res
            
            print("-"*70)
            print("idx: {:3}".format(idx))
            print("                                   \t{}".format(features))
            print()
            print("Predicted(y_pred)  :\t{}".format(predicted.detach().numpy()[0]))
            print("Target(y^)               :\t{}".format(target[0]))
            print()
            print("Last Price(y-1)        :\t{}".format(previous[0]))
            print()
            print("(y^ - (y_pred))      => {}".format(pred_res))
            print("(y^ - (y-1))             => {}".format(base_res))
            print("-"*70)
            
            return (base_res - pred_res) / base_res
            
    if type(idx_list) == int:
        
        res = compare_and_print(df, df_label, idx_list, features, non_label_index, bt_size, verbose)
        
        print()
        print("Comparison to Baseline: \n")
        print(res.item())
        print()
        print()
        
        return 
        
    elif type(idx_list) == list:
        
        benchmark = []
   
        for idx in idx_list:
            res = compare_and_print(df, df_label, idx, features, non_label_index, bt_size, verbose)
            benchmark.append(res)
       
    print()
    print("Average Comparison to Baseline: \n")
    print((sum(benchmark)/len(benchmark)).item())
    print()
    print()
    
    return 
            
# calculate baseline (not working)
def mse_copy_last_baseline(df, df_label, bt_size, scale=100):
    mse = []
    for i in range(df.shape[0]):
        predicted = df[i][-1]
        target = df_label[i]
        loss = ((predicted - target) **2).sum()
        mse.append(loss)
    return scale * (sum(mse) / (len(mse) // bt_size))