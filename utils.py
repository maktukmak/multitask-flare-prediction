
import os
import pandas as pd

def print_ar_stats(dataset):
    print("Harps total:", len(os.listdir(dataset.path_harp)))
    print("Harps removed for noaa:", dataset.cnt_harp_noaa)
    print("Harps without flares:", dataset.cnt_harp_wo_flare)
    print("Harps used:", len(os.listdir(dataset.path_harp)) - dataset.cnt_harp_noaa - dataset.cnt_harp_wo_flare)
    print("Harps length average:", dataset.harp_lengths.mean())
    print("Harps length min:", dataset.harp_lengths.min())
    print("Harps length max:", dataset.harp_lengths.max())
    
    
def print_flare_stats(dataset):
    print("Flares total:", len(dataset.valid_events) + dataset.cnt_flare_m_sharp)
    print("Flares with missing:", dataset.cnt_flare_m_sharp)
    print("Flares used train:", len(dataset.valid_events_train))
    print("Flares used test:", len(dataset.valid_events_test))
    print("Flares used train-B:", len([idx for idx, element in enumerate(dataset.valid_events_train) if element[1][:1] == 'B']))
    print("Flares used train-C:", len([idx for idx, element in enumerate(dataset.valid_events_train) if element[1][:1] == 'C']))
    print("Flares used train-M:", len([idx for idx, element in enumerate(dataset.valid_events_train) if element[1][:1] == 'M']))
    print("Flares used train-X:", len([idx for idx, element in enumerate(dataset.valid_events_train) if element[1][:1] == 'X']))
    print("Flares used test-B:", len([idx for idx, element in enumerate(dataset.valid_events_test) if element[1][:1] == 'B']))
    print("Flares used test-C:", len([idx for idx, element in enumerate(dataset.valid_events_test) if element[1][:1] == 'C']))
    print("Flares used test-M:", len([idx for idx, element in enumerate(dataset.valid_events_test) if element[1][:1] == 'M']))
    print("Flares used test-X:", len([idx for idx, element in enumerate(dataset.valid_events_test) if element[1][:1] == 'X']))
        
def print_intersect_flare_stats(dataset):
    print("Flares total:", len(dataset.valid_events_train_intersect))
    print("Flares B:", len([idx for idx, element in enumerate(dataset.valid_events_train_intersect) if element[2][:1] == 'B']))
    print("Flares C:", len([idx for idx, element in enumerate(dataset.valid_events_train_intersect) if element[2][:1] == 'C']))
    print("Flares M:", len([idx for idx, element in enumerate(dataset.valid_events_train_intersect) if element[2][:1] == 'M']))
    print("Flares X:", len([idx for idx, element in enumerate(dataset.valid_events_train_intersect) if element[2][:1] == 'X']))
    
def small_data(dataset):
    ind_BC = [idx for idx, element in enumerate(dataset.valid_events_train) if element[1][:1] == 'B' or element[1][:1] == 'C']
    ind_MX = [idx for idx, element in enumerate(dataset.valid_events_train) if element[1][:1] == 'M' or element[1][:1] == 'X']
    ind = ind_MX[:16] + ind_BC[:16]
    dataset.valid_events_train = [dataset.valid_events_train[i] for i in ind]
    ind_BC = [idx for idx, element in enumerate(dataset.valid_events_test) if element[1][:1] == 'B' or element[1][:1] == 'C']
    ind_MX = [idx for idx, element in enumerate(dataset.valid_events_test) if element[1][:1] == 'M' or element[1][:1] == 'X']
    ind = ind_MX[:16] + ind_BC[:16]
    dataset.valid_events_test = [dataset.valid_events_test[i] for i in ind]
    

    
    
