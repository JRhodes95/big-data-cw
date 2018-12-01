from __future__ import print_function

from shallow import ShallowLearner
from deep import DeepLearner

import pandas as pd
import numpy as np
import plots
import os

def main():
    """Displays a CLI to display the execution process and results of the various classifiers."""
    print("This is the 'Big Data' summative assignment for Z0954757.")
    print()
    # Perform analysis for the Shallow Learning Intitial Investiagtion
    shallow_initial = raw_input(
    "Would you like to run the initial investigations for the shallow learning approaches (Estimated time to complete: 3 minutes)? (y/n)"
                                )
    if 'y' in shallow_initial.lower():
        # Create an instance of the ShallowLearner class
        shall = ShallowLearner()
        # Get the data for use in the shallow appraches
        shall.get_data(os.path.join('datasets','news_ds.csv'))
        # Try the first approach
        first_results = shall.first_approach()
        print(first_results)
        # Try the second approach
        second_results = shall.second_approach()
        print(second_results)
        # Try the third approach
        third_results = shall.third_approach()
        print(third_results)
        # Try the fourth approach
        fourth_results = shall.fourth_approach()
        print(fourth_results)


    # Perform analysis for the Shallow Learning Further Investigations
    shallow_further = raw_input(
    "Would you like to run the further investigations for the shallow learning approaches? (y/n)"
                                )
    if 'y' in shallow_further.lower():
        load_data = raw_input(
        "Type 'load' to load pre-existing data or nothing to regenerate the data (Estimated time to regenerate: 90 minutes)"
                            )
        if 'load' in load_data.lower():
            #Load data from csv files.
            plots.plot_grid_search(os.path.join('saves','ThirdApproachVariations.csv'), ' Third Approach - TF-IDF Grid Search Optimisation')
            plots.plot_grid_search(os.path.join('saves','FourthApproachVariations.csv'), ' Fourth Approach - N-gram (1,2) Grid Search Optimisation')
        else:
            print("Regenerating data.")

            # Create an instance of the ShallowLearner class
            shall = ShallowLearner()
            # Get the data for use in the shallow appraches
            shall.get_data(os.path.join('datasets','news_ds.csv'))
            # Create arrays of test values for splits and max features.
            splits = np.arange(0.2, 1, 0.2)
            max_feats = np.arange(1000, 21000, 2000)

            print("Test splits: ", splits)
            print("Test maximum features: ", max_feats)

            # Intialise a dictionary to collect the results.
            third_results_dict = {
                'splits' : [],
                'no feats' : [],
                'Accuracy': [],
                'Precision' : [],
                'Recall':[],
                'F1':[]
            }

            print("Varying splits and max features for approach three.")
            for test_split in splits:
                print("Testing at split: ", test_split)
                for features in max_feats:
                    print("Testing at max features: ", features)
                    results = shall.third_approach(split=test_split, no_features=features)
                    third_results_dict['splits'].append(test_split)
                    third_results_dict['no feats'].append(features)
                    third_results_dict['Accuracy'].append(results['Accuracy'])
                    third_results_dict['Precision'].append(results['Precision'])
                    third_results_dict['Recall'].append(results['Recall'])
                    third_results_dict['F1'].append(results['F1'])

            third_results_df = pd.DataFrame(third_results_dict)
            third_results_df.to_csv(os.path.join('saves','ThirdApproachVariationsRegen.csv'))

            # Vary n-gram format in approach four
            print("Varying n-gram range for approach four.")
            n_gram_ranges = [(1,1),(2,2), (3,3), (1,2), (1,3)]
            fourth_n_gram_results_dict = {
                'n_gram_range' : [],
                'Accuracy': [],
                'Precision' : [],
                'Recall':[],
                'F1':[]
            }

            for n_range in n_gram_ranges:
                print("Testing n gram range: ", n_range)
                results = shall.fourth_approach(n_range)
                fourth_n_gram_results_dict['n_gram_range'].append(n_range)
                fourth_n_gram_results_dict['Accuracy'].append(results['Accuracy'])
                fourth_n_gram_results_dict['Precision'].append(results['Precision'])
                fourth_n_gram_results_dict['Recall'].append(results['Recall'])
                fourth_n_gram_results_dict['F1'].append(results['F1'])

            fourth_n_gram_results_df = pd.DataFrame(fourth_n_gram_results_dict)
            fourth_n_gram_results_df.to_csv(os.path.join('saves','FourthApproachNGramsRegen.csv'))

            # Intialise a dictionary to collect the results.
            fourth_results_dict = {
                'splits' : [],
                'no feats' : [],
                'Accuracy': [],
                'Precision' : [],
                'Recall':[],
                'F1':[]
            }

            print("Varying splits and max features for approach three.")
            for test_split in splits:
                print("Testing at split: ", test_split)
                for features in max_feats:
                    print("Testing at max features: ", features)
                    results = shall.fourth_approach(n_range=(1,2), split=test_split, no_features=features)
                    fourth_results_dict['splits'].append(test_split)
                    fourth_results_dict['no feats'].append(features)
                    fourth_results_dict['Accuracy'].append(results['Accuracy'])
                    fourth_results_dict['Precision'].append(results['Precision'])
                    fourth_results_dict['Recall'].append(results['Recall'])
                    fourth_results_dict['F1'].append(results['F1'])

            fourth_results_df = pd.DataFrame(fourth_results_dict)
            fourth_results_df.to_csv(os.path.join('saves','FourthApproachVariationsRegen.csv'))


            plots.plot_grid_search(os.path.join('saves','ThirdApproachVariationsRegen.csv'), ' Third Approach - TF-IDF Grid Search Optimisation')
            plots.plot_grid_search(os.path.join('saves','FourthApproachVariationsRegen.csv'), ' Fourth Approach - N-gram (1,2) Grid Search Optimisation')
    # Perform analysis for the Deep Learning Investigation
    deep_analysis = raw_input(
    "Would you like to run the analysis for the deep learning approach? (y/n)"
                                )
    if 'y' in deep_analysis.lower():
        # Create an instance of the DeepLearner class
        deep = DeepLearner()
        # Get the data for the deep approach
        deep.get_data(os.path.join('datasets','news_ds.csv'))
        # Try the LSTM approach
        results = deep.lstm_approach()
        print(results)

    print("Closing CLI.")



if __name__ == "__main__":
    main()
