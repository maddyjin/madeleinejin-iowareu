#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy
from scipy.stats import shapiro, anderson, stats, pearsonr, zscore, ttest_ind
import statistics
import matplotlib.pyplot as plt
import math
import seaborn as sns
sns.set()

# Not resampled
icu291visits = pd.read_csv('icu29.1.csv')
icu291shifts = pd.read_csv('icu291shifts.csv')
icu292visits = pd.read_csv('icu29.2.csv')
icu292shifts = pd.read_csv('icu292shifts.csv')
icu293visits = pd.read_csv('icu29.3.csv')
icu293shifts = pd.read_csv('icu293shifts.csv')
icu294visits = pd.read_csv('icu29.4.csv')
icu294shifts = pd.read_csv('icu294shifts.csv')
icu295visits = pd.read_csv('icu29.5.csv')
icu295shifts = pd.read_csv('icu295shifts.csv')
icu296visits = pd.read_csv('icu29.6.csv')
icu296shifts = pd.read_csv('icu296shifts.csv')
icu297visits = pd.read_csv('icu29.7.csv')
icu297shifts = pd.read_csv('icu297shifts.csv')
icu298visits = pd.read_csv('icu29.8.csv')
icu298shifts = pd.read_csv('icu298shifts.csv')
icu299visits = pd.read_csv('icu29.9.csv')
icu299shifts = pd.read_csv('icu299shifts.csv')
icu2910visits = pd.read_csv('icu29.10.csv')
icu2910shifts = pd.read_csv('icu2910shifts.csv')
visits_df1 = [icu291visits, icu292visits, icu293visits, icu294visits, icu295visits, icu296visits, \
              icu297visits, icu298visits, icu299visits, icu2910visits]
shifts_df1 = [icu291shifts, icu292shifts, icu293shifts, icu294shifts, icu295shifts, icu296shifts, \
              icu297shifts, icu298shifts, icu299shifts, icu2910shifts]

# Resampled
icu29avisits = pd.read_csv('icu29.a.csv')
icu29ashifts = pd.read_csv('icu29ashifts.csv')
icu29bvisits = pd.read_csv('icu29.b.csv')
icu29bshifts = pd.read_csv('icu29bshifts.csv')
icu29cvisits = pd.read_csv('icu29.c.csv')
icu29cshifts = pd.read_csv('icu29cshifts.csv')
icu29dvisits = pd.read_csv('icu29.d.csv')
icu29dshifts = pd.read_csv('icu29dshifts.csv')
icu29evisits = pd.read_csv('icu29.e.csv')
icu29eshifts = pd.read_csv('icu29eshifts.csv')
icu29fvisits = pd.read_csv('icu29.f.csv')
icu29fshifts = pd.read_csv('icu29fshifts.csv')
icu29gvisits = pd.read_csv('icu29.g.csv')
icu29gshifts = pd.read_csv('icu29gshifts.csv')
icu29hvisits = pd.read_csv('icu29.h.csv')
icu29hshifts = pd.read_csv('icu29hshifts.csv')
icu29ivisits = pd.read_csv('icu29.i.csv')
icu29ishifts = pd.read_csv('icu29ishifts.csv')
icu29jvisits = pd.read_csv('icu29.j.csv')
icu29jshifts = pd.read_csv('icu29jshifts.csv')
visits_df2 = [icu29avisits, icu29bvisits, icu29cvisits, icu29dvisits, icu29evisits, icu29fvisits, \
              icu29gvisits, icu29hvisits, icu29ivisits, icu29jvisits]
shifts_df2 = [icu29ashifts, icu29bshifts, icu29cshifts, icu29dshifts, icu29eshifts, icu29fshifts, \
              icu29gshifts, icu29hshifts, icu29ishifts, icu29jshifts]

def normality(ls):
    """
    Shapiro-Wilk test and Anderson-Darling test.
    
    Arguments:
        ls: list of numbers
    """
    try:
        sw, ad = shapiro(ls).pvalue, anderson(ls).statistic
    except:
        sw, ad = 0, 0
    print(f'Shapiro-Wilk p-value: {np.nanmean(sw)}, Anderson-Darling statistic: {np.nanmean(ad)}')
    
def t_test(ls1, ls2):
    """
    Runs either a two-sample t-test on ls1 and ls2 and returns p-value.
    
    Arguments:
        ls1: list of numbers from first schedule
        ls2: list of numbers from second schedule
    Returns:
        p_value: p-value from t-test
    """
    _, p_value = stats.ttest_ind(ls1, ls2)
    if not math.isnan(p_value):
        return p_value
    else:
        print('error calculating p-value')
        return 0
    
def plot_hist(data1, data2, title, xlabel, ylabel, p_value=None):
    """
    Plots a histogram where data1 is a list of values from schedule 1 and is in blue and data2 is a list
    of values from schedule 2 and is in red. Helps visualize the comparison between the two schedules
    as a supplement to a p-value.
    
    Arguments:
    data1: List of numbers representative of an attribute of schedule 1
    data2: List of numbers representative of an attribute of schedule 2
    title: Title of hisogram
    xlabel: Label for x-axis of histogram
    ylabel: Label for x-axis of histogram
    p_value: p-value for t-test (default = None)
    """
    plt.hist(data1, alpha=0.5, bins=7, color='blue', label='Trace 1: Real Data')
    plt.hist(data2, alpha=0.5, bins=7, color='red', label='Trace 2: Remixed Data')
    if p_value is not None:
        title += f'\n p-value: {p_value}'
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
# Similarity matrix heat map compares derivative values from two
# schedules: used to compare, e.g., visit durations, # visits per
# shift, # unique HCWs etc etc.
def stability(values1, values2, title):
    """
    Compares each data point using a similarity matrix.
    
    Returns:
        df: Similarity matrix heatmap
    """
    num_values1 = len(values1)
    num_values2 = len(values2)

    # Initialize the similarity matrix with zeros
    similarity_matrix = np.zeros((num_values1, num_values2), dtype=float)
    
    # Fill in the similarity matrix
    for i in range(num_values1):
        for j in range(num_values2):
            similarity_matrix[i, j] = np.sqrt(np.sum((values1[i] - values2[j])**2))

    # Convert the similarity matrix to a DataFrame
    df = pd.DataFrame(similarity_matrix, index=values1, columns=values2)

    # Create the heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='viridis')
    plt.title(f"Similarity Matrix Heatmap: {title}")
    plt.xlabel("Values 2")
    plt.ylabel("Values 1")
    plt.show()

    return df

# Index value: smaller sums are better, meaning they indicate more
# similar value sets. 
def z_score(values1, values2):
    """
    Calculate the z score of the values in the similarity matrix.

    Returns:
        z_score: z score of the values in the similarity matrix
    """
    similarity_matrix = np.zeros((len(values1), len(values2)), dtype=float)
    for i in range(len(values1)):
        for j in range(len(values2)):
            similarity_matrix[i, j] = np.abs(values1[i] - values2[j])

    # Calculate stability score as the sum of all values in the similarity matrix
    score = similarity_matrix.sum()
    
    # Calculate z-score
    ls = []
    for i in range(len(values1)):
        for j in range(len(values2)):
            ls.append(similarity_matrix[i, j])
    
    z_score = f'z-score={statistics.median(scipy.stats.zscore(ls))}'

    return z_score

def jtid_tests(values_1, values_2):
    """
    Takes values_1 and values_2, prepares them for tests to be ran, and runs the tests.
    
    Arguments:
        values_1 (Dictionary): Key is JTID, value is list of values from schedule 1
        values_2 (Dictionary): Key is JTID, value is list of values from schedule 2
        
    Returns:
        p_value_score (float): Overall p-value weighted based on number of HCWs for each JTID
        overall_z_score (float): Overall z-score weighted based on number of HCWs for each JTID
    """
    # Ensure all lists have equal values by padding with zeros if necessary
    # Calculate the maximum length of all lists in values_1 and values_2
    max_length = max(max(map(len, values_1.values())), max(map(len, values_2.values())))

    # Pad each list in values_1 and values_2 with zeros to make their lengths equal to max_length
    for key, value in values_1.items():
        value.extend([0] * (max_length - len(value)))
    for key, value in values_2.items():
        value.extend([0] * (max_length - len(value)))
        
    # Create an empty dictionary to store the t-test and z-score results
    tests = {}
    # Iterate through the keys in 'values_1' and 'values_2' and perform the t-test
    for key in values_1.keys():
        # Perform t-test on corresponding values for 'jtid' in values_1 and values_2
        t_statistic, p_value = ttest_ind(values_1[key], values_2[key])
        # Calculate z-score for the two lists
        arr1 = np.array(values_1[key])
        arr2 = np.array(values_2[key])
        combined_array = np.concatenate((arr1, arr2))
        z_scores = zscore(combined_array)
        # Get the z-score for the second list (values_2)
        z_score_value = z_scores[len(arr1)]

        # Check if values iare NaN, and if so, assign 1 to the t_tests dictionary, otherwise, assign value
        tests[key] = [1 if math.isnan(p_value) else p_value,
                      0 if math.isnan(z_score_value) else z_score_value]
        
    # Dictionary where the key is the JTID and the values is the sum of the number of HCWs in both schedules
    # Used to determine weight of overall score, more HCWs --> more weight in score
    num_hcws_jtid = {key: sum(values_1[key]) + sum(values_2[key]) for key in values_1}
    total_sum = sum(num_hcws_jtid.values())

    # Weight based on number of HCWs for each JTID to calculate overall score for p-values
    p_value_score = sum(tests[key][0] * num_hcws_jtid[key] for key in tests if tests[key][0] != 0)
    p_value_score = p_value_score / total_sum
    print(f'Overall P-Value Score Weighted by Number of HCWs: {p_value_score}')

    # Weight based on number of HCWs for each JTID to calculate overall score for z-scores
    overall_z_score = sum(tests[key][1] * num_hcws_jtid[key] for key in tests if tests[key][1] != 0)
    overall_z_score = overall_z_score / total_sum
    print(f'Overall Z-Score Weighted by Number of HCWs: {overall_z_score}')

    return p_value_score, overall_z_score


# In[2]:


# Test1: uses machinery above to compare NUMBER OF VISITS TOTAL
def visits_validation():
    """
    Executes a two-sample t-test to compare the number of visits total during 
    between schedule 1 and schedule 2.
    
    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    visits_ls_1 = []
    for index in range(1, len(visits_df1)+1):
        # Check if the DataFrame is not empty before appending the length
        if not visits_df1[index-1].empty:
            visits_ls_1.append(len(visits_df1[index-1]))  
    # Calculate the normality and z score for Schedule 1
    normality(ls=visits_ls_1)
    stability(values1=visits_ls_1, values2=visits_ls_1, title=f'Schedule 1, \
    {z_score(values1=visits_ls_1, values2=visits_ls_1)}')
    z_score_1 = z_score(values1=visits_ls_1, values2=visits_ls_1)
                
    # Schedule 2
    visits_ls_2 = []
    for index in range(1, len(visits_df2)+1):
        # Check if the DataFrame is not empty before appending the length
        if not visits_df2[index-1].empty:
            visits_ls_2.append(len(visits_df2[index-1]))
    # Calculate the normality and z score for Schedule 2
    normality(ls=visits_ls_2)
    stability(values1=visits_ls_2, values2=visits_ls_2, title=f'Schedule 2, \
    {z_score(values1=visits_ls_2, values2=visits_ls_2)}')
    z_score_2 = z_score(values1=visits_ls_2, values2=visits_ls_2)
    
    # Calculate the p-value using a t-test
    p_value = t_test(ls1=visits_ls_1, ls2=visits_ls_2)
    
    # Calculate and display the z score between Schedule 1 and Schedule 2
    visits_z_score = z_score(values1=visits_ls_1, values2=visits_ls_2)
    stability(values1=visits_ls_1, values2=visits_ls_2, title=f'Schedule 1 vs. Schedule 2, \
    {z_score(values1=visits_ls_1, values2=visits_ls_2)}')
    visits_z_score = z_score(values1=visits_ls_1, values2=visits_ls_2)
    
    # Plot histograms to compare the distributions
    plot_hist(visits_ls_1, visits_ls_2, 'Number of visits total during the schedule', \
              'Number of visits', 'Frequency', p_value=p_value)
    
    return p_value, visits_z_score

# Perform the visits validation and get the p-value and lists of visits for both schedules
p_value_visits, z_score_visits = visits_validation()

def visits_by_jtid():
    """
    Sorts visits dataframes by JTID and performs t-test on the number of unique 'vid's with each 'jtid'.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the count of unique 'vid's for each 'jtid' as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df1
    for df in visits_df1:
        # Group by 'jtid' and count unique 'vid's in each group
        for jtid, count in df.groupby('jtid')['vid'].nunique().items():
            # Append the 'vid' count to the corresponding 'jtid' key in values_1
            values_1[jtid].append(count)

    # Schedule 2
    # Initialize a dictionary to store the count of unique 'vid's for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df2
    for df in visits_df2:
        # Group by 'jtid' and count unique 'vid's in each group
        for jtid, count in df.groupby('jtid')['vid'].nunique().items():
            # Append the 'vid' count to the corresponding 'jtid' key in values_2
            values_2[jtid].append(count)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_visits, overall_z_score_visits = visits_by_jtid()


# In[3]:


# Test2: uses machinery above to compare NUMBER OF VISITS PER SHIFT
def visits_shift_validation():
    """
    Executes a two-sample t-test to compare the number of visits per shift during 
    between schedule 1 and schedule 2.

    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    visits_shift_ls_1 = []
    for index in range(1, len(visits_df1)+1):
        # Check if both visits_df1 and shifts_df1 are not empty before calculating visits per shift
        if not visits_df1[index-1].empty and not shifts_df1[index-1].empty:
            visits_shift_ls_1.append(len(visits_df1[index-1]) / len(shifts_df1[index-1]))   
    normality(ls=visits_shift_ls_1)
    # Calculate and display the z score for Schedule 1 based on visits per shift
    stability(values1=visits_shift_ls_1, values2=visits_shift_ls_1, \
              title=f'Schedule 1, {z_score(values1=visits_shift_ls_1, values2=visits_shift_ls_1)}')
        
    # Schedule 2
    visits_shift_ls_2 = []
    for index in range(1, len(visits_df2)+1):
        # Check if both visits_df2 and shifts_df2 are not empty before calculating visits per shift
        if not visits_df2[index-1].empty and not shifts_df2[index-1].empty:
            visits_shift_ls_2.append(len(visits_df2[index-1]) / len(shifts_df2[index-1]))
    normality(ls=visits_shift_ls_2)
    # Calculate and display the z score for Schedule 2 based on visits per shift
    stability(values1=visits_shift_ls_2, values2=visits_shift_ls_2, \
              title=f'Schedule 2, {z_score(values1=visits_shift_ls_2, values2=visits_shift_ls_2)}')
    
    # Calculate the p-value using a t-test for comparing visits per shift between schedules
    p_value = t_test(ls1=visits_shift_ls_1, ls2=visits_shift_ls_2)
    
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on visits per shift
    visits_shift_z_score = z_score(values1=visits_shift_ls_1, values2=visits_shift_ls_2)
    stability(values1=visits_shift_ls_1, values2=visits_shift_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, \
              {z_score(values1=visits_shift_ls_1, values2=visits_shift_ls_2)}')
    visits_shift_z_score = z_score(values1=visits_shift_ls_1, values2=visits_shift_ls_2)

    # Plot histograms to compare the distributions of visits per shift for both schedules
    plot_hist(visits_shift_ls_1, visits_shift_ls_2, 'Number of visits per shift', 'Number of visits', \
              'Frequency', p_value=p_value)
    
    return p_value, visits_shift_z_score

# Perform the visits per shift validation and get the p-value and lists of visits per shift for schedules
p_value_visits_shift, z_score_visits_shift = visits_shift_validation()

def visits_shift_by_jtid():
    """
    Sorts visits and shifts dataframes by JTID and performs t-test on the number of visits per shift

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the count of unique 'vid's for each 'jtid' as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df1 and shifts_df1 simultaneously
    for visits_df, shifts_df in zip(visits_df1, shifts_df1):
        # Group by 'jtid' and count unique 'vid's in each group
        for jtid, count in visits_df.groupby('jtid')['vid'].nunique().items():
            # Get the number of rows in the corresponding DataFrame in shifts_df1
            num_rows_shifts = len(shifts_df[shifts_df['jtid'] == jtid])
            # Check if the 'jtid' exists in shifts_df1
            if num_rows_shifts > 0:
                # Calculate the ratio of 'vid' count to the number of rows in shifts_df1 for the 'jtid'
                ratio = count / num_rows_shifts
                # Append the calculated ratio to the corresponding 'jtid' key in values_1
                values_1[jtid].append(ratio)

    # Schedule 2
    # Initialize a dictionary to store the count of unique 'vid's for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df2 and shifts_df2 simultaneously
    for visits_df, shifts_df in zip(visits_df2, shifts_df2):
        # Group by 'jtid' and count unique 'vid's in each group
        for jtid, count in visits_df.groupby('jtid')['vid'].nunique().items():
            # Get the number of rows in the corresponding DataFrame in shifts_df1
            num_rows_shifts = len(shifts_df[shifts_df['jtid'] == jtid])
            # Check if the 'jtid' exists in shifts_df2
            if num_rows_shifts > 0:
                # Calculate the ratio of 'vid' count to the number of rows in shifts_df2 for the 'jtid'
                ratio = count / num_rows_shifts
                # Append the calculated ratio to the corresponding 'jtid' key in values_2
                values_2[jtid].append(ratio)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_visits_shift, overall_z_score_visits_shift = visits_shift_by_jtid()


# In[4]:


# Test3: uses machinery above to compare NUMBER OF SHIFTS
def shifts_validation():
    """
    Executes a two-sample t-test to compare the number of shifts total during the schedule between 
    schedule 1 and schedule 2.
    
    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    shifts_ls_1 = []
    for index in range(1, len(visits_df1)+1):
        # Check if the shifts DataFrame is not empty before appending the number of shifts
        if not shifts_df1[index-1].empty:
             shifts_ls_1.append(len(shifts_df1[index-1]))
    normality(ls=shifts_ls_1)
    # Calculate and display the z score for Schedule 1 based on total shifts
    stability(values1=shifts_ls_1, values2=shifts_ls_1, \
              title=f'Schedule 1, {z_score(values1=shifts_ls_1, values2=shifts_ls_1)}')
                
    # Schedule 2
    shifts_ls_2 = []
    for index in range(1, len(visits_df2)+1):
        # Check if the shifts DataFrame is not empty before appending the number of shifts
        if not shifts_df2[index-1].empty:
            shifts_ls_2.append(len(shifts_df2[index-1]))         
    normality(ls=shifts_ls_2)
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on total shifts
    stability(values1=shifts_ls_1, values2=shifts_ls_2, \
              title=f'Schedule 2, {z_score(values1=shifts_ls_1, values2=shifts_ls_2)}')
    
    # Calculate the p-value using a t-test for comparing the total number of shifts between schedules
    p_value = t_test(ls1=shifts_ls_1, ls2=shifts_ls_2)
    
    # Calculate and display the z score for Schedule 2 based on total shifts
    shifts_z_score = z_score(values1=shifts_ls_2,values2=shifts_ls_2)
    stability(values1=shifts_ls_2, values2=shifts_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, {z_score(values1=shifts_ls_2,values2=shifts_ls_2)}')

    # Calculate and display the z score between Schedule 1 and Schedule 2 based on total shifts
    stability(values1=shifts_ls_1, values2=shifts_ls_2, \
              title=f'Schedule 2, {z_score(values1=shifts_ls_1, values2=shifts_ls_2)}')
    z_score_shifts = z_score(values1=shifts_ls_1, values2=shifts_ls_2)
    
    # Plot histograms to compare the distributions of total shifts for both schedules
    plot_hist(shifts_ls_1, shifts_ls_2, 'Number of shifts total during the schedule', \
              'Number of shifts', 'Frequency', p_value=p_value)
    
    return p_value, z_score_shifts

# Perform the shifts validation and get the p-value and lists of total shifts for both schedules
p_value_shifts, z_score_shifts = shifts_validation()

def shifts_by_jtid():
    """
    Sorts shifts dataframes by JTID and performs t-test on the number of shifts.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the count of shifts as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df1
    for shifts_df in shifts_df1:
        # Group by 'jtid' and count shifts in each group
        for jtid, count in shifts_df.groupby('jtid').size().items():
            # Append the shifts count to the corresponding 'jtid' key in values_1
            values_1[jtid].append(count)

    # Schedule 2
    # Initialize a dictionary to store the count of unique 'vid's for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df2
    for shifts_df in shifts_df2:
        # Group by 'jtid' and count shifts in each group
        for jtid, count in shifts_df.groupby('jtid').size().items():
            # Append the shifts count to the corresponding 'jtid' key in values_2
            values_2[jtid].append(count)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_shifts, overall_z_score_shifts = shifts_by_jtid()


# In[5]:


# Test4: uses machinery above to compare TOTAL VISIT TIME 
def dur_visits_validation():
    """
    Executes a two-sample t-test to compare the average visit length between schedule 1 and schedule 2.
    
    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    dur_visits_ls_1 = []
    for index in range(1, len(visits_df1)+1):
        # Check if the visits DataFrame is not empty before calculating the average visit length
        if not visits_df1[index-1].empty:
            dur_visits_ls_1.append(visits_df1[index-1]['duration'].mean())
    normality(ls=dur_visits_ls_1)
    # Calculate and display the z score for Schedule 1 based on average visit length
    stability(values1=dur_visits_ls_1, values2=dur_visits_ls_1, \
              title=f'Schedule 1, {z_score(values1=dur_visits_ls_1, values2=dur_visits_ls_1)}')
       
    # Schedule 2
    dur_visits_ls_2 = []
    for index in range(1, len(visits_df2)+1):
        # Check if the visits DataFrame is not empty before calculating the average visit length
        if not visits_df2[index-1].empty:
            dur_visits_ls_2.append(visits_df2[index-1]['duration'].mean())
    normality(ls=dur_visits_ls_2)
    # Calculate and display the z score for Schedule 2 based on average visit length
    stability(values1=dur_visits_ls_2, values2=dur_visits_ls_2, \
              title=f'Schedule 2, {z_score(values1=dur_visits_ls_2, values2=dur_visits_ls_2)}')
    
    # Calculate the p-value using a t-test for comparing the average visit length between schedules
    p_value = t_test(ls1=dur_visits_ls_1, ls2=dur_visits_ls_2)
    
    stability(values1=dur_visits_ls_1, values2=dur_visits_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, \
              {z_score(values1=dur_visits_ls_1, values2=dur_visits_ls_2)}')
    dur_visits_z_score = z_score(values1=dur_visits_ls_1, values2=dur_visits_ls_2)
    
    # Plot histograms to compare the distributions of average visit length for both schedules
    plot_hist(dur_visits_ls_1, dur_visits_ls_2, 'Average duration of visits', \
              'Visit Duration', 'Frequency', p_value=p_value)
    
    return p_value, dur_visits_z_score

# Perform the average visit length validation and get the p-value and lists of average visit lengths for 
# both schedules
p_value_dur_visits, dur_visits_z_score = dur_visits_validation()

def dur_visits_by_jtid():
    """
    Sorts visits dataframes by JTID and performs t-test on the duration of visits,

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the sum of 'duration' for each 'jtid' as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df1
    for df in visits_df1:
        # Group by 'jtid' and calculate the sum of 'duration' in each group
        for jtid, duration_sum in df.groupby('jtid')['duration'].sum().items():
            # Append the 'duration' sum to the corresponding 'jtid' key in values_1
            values_1[jtid].append(duration_sum)

    # Schedule 2
    # Initialize a dictionary to store the count of unique 'vid's for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df2
    for df in visits_df2:
        # Group by 'jtid' and calculate the sum of 'duration' in each group
        for jtid, duration_sum in df.groupby('jtid')['duration'].sum().items():
            # Append the 'duration' sum to the corresponding 'jtid' key in values_2
            values_2[jtid].append(duration_sum)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_dur_visits, overall_z_score_dur_visits = dur_visits_by_jtid()


# In[6]:


# Test5: uses machinery above to compare PERCENTAGE OF VISIT TIME OVER SHIFT TIME
def percent_worked_validation():
    """
    Executes a two-sample t-test to compare the average percentage of time spent on visits during shift 
    between schedule 1 and schedule 2.
    
    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    percent_worked_ls_1 = []
    for index in range(1, len(visits_df1)+1):
        # Check if both visits_df1 and shifts_df1 are not empty before calculating the average percentage
        # of time worked
        if not visits_df1[index-1].empty and not shifts_df1[index-1].empty:
            percent_worked_ls_1.append(visits_df1[index-1]['duration'].sum() / \
                                       shifts_df1[index-1]['duration'].sum()) 
    normality(ls=percent_worked_ls_1)
    # Calculate and display the z score for Schedule 1 based on average percentage of time worked
    stability(values1=percent_worked_ls_1, values2=percent_worked_ls_1, \
              title=f'Schedule 1, \
              {z_score(values1=percent_worked_ls_1, values2=percent_worked_ls_1)}')
           
    # Schedule 2
    percent_worked_ls_2 = []
    for index in range(1, len(visits_df2)+1):
        # Check if both visits_df2 and shifts_df2 are not empty before calculating the average percentage
        # of time worked
        if not visits_df2[index-1].empty and not shifts_df2[index-1].empty:
            percent_worked_ls_2.append(visits_df2[index-1]['duration'].sum() / \
                                       shifts_df2[index-1]['duration'].sum())   
    normality(ls=percent_worked_ls_2)
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on average 
    # percentage of time worked
    stability(values1=percent_worked_ls_1, values2=percent_worked_ls_2, \
              title=f'Schedule 2, \
              {z_score(values1=percent_worked_ls_2, values2=percent_worked_ls_2)}')
    
    # Calculate the p-value using a t-test for comparing the average percentage of time worked between 
    # schedules
    p_value = t_test(ls1=percent_worked_ls_1, ls2=percent_worked_ls_2)
    
    # Calculate and display the z score for Schedule 2 based on average percentage of time worked
    percent_worked_z_score = z_score(values1=percent_worked_ls_1, values2=percent_worked_ls_2)
    stability(values1=percent_worked_ls_1, values2=percent_worked_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, \
              {z_score(values1=percent_worked_ls_1, values2=percent_worked_ls_2)}')
    percent_worked_z_score = z_score(values1=percent_worked_ls_1, values2=percent_worked_ls_2)
    
    # Plot histograms to compare the distributions of the average percentage of time worked for both 
    # schedules
    plot_hist(percent_worked_ls_1, percent_worked_ls_2, \
              'Avg percent of time spent on visits during shift', \
              'Percent', 'Frequency', p_value=p_value)
    
    return p_value, percent_worked_z_score

# Perform the average percentage of time worked validation and get the p-value and lists of average 
# percentage of time worked for both schedules
p_value_percent_worked, percent_worked_z_score = percent_worked_validation()

def percent_worked_by_jtid():
    """
    Sorts visits and shifts dataframes by JTID and performs t-test on the percent of each shift that was
    spent on visits.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the count of unique 'vid's for each 'jtid' as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df1 and shifts_df1 simultaneously
    for visits_df, shifts_df in zip(visits_df1, shifts_df1):
        # Group by 'jtid' and calculate the total duration of visits in each group
        for jtid, visit_duration_sum in visits_df.groupby('jtid')['duration'].sum().items():
            # Get the total duration of shifts for the 'jtid'
            total_shift_duration = shifts_df[shifts_df['jtid'] == jtid]['duration'].sum()
            # Check if the 'jtid' exists in shifts_df1 and if the total_shift_duration is not zero
            if total_shift_duration > 0:
                # Calculate the percentage of time spent on visits during each shift for the 'jtid'
                percentage_worked = visit_duration_sum / total_shift_duration
                # Append the calculated percentage to the corresponding 'jtid' key in values_1
                values_1[jtid].append(percentage_worked)

    # Schedule 2
    # Initialize a dictionary to store the count of unique 'vid's for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df2 and shifts_df2 simultaneously
    for visits_df, shifts_df in zip(visits_df2, shifts_df2):
        # Group by 'jtid' and calculate the total duration of visits in each group
        for jtid, visit_duration_sum in visits_df.groupby('jtid')['duration'].sum().items():
            # Get the total duration of shifts for the 'jtid'
            total_shift_duration = shifts_df[shifts_df['jtid'] == jtid]['duration'].sum()
            # Check if the 'jtid' exists in shifts_df1 and if the total_shift_duration is not zero
            if total_shift_duration > 0:
                # Calculate the percentage of time spent on visits during each shift for the 'jtid'
                percentage_worked = visit_duration_sum / total_shift_duration
                # Append the calculated percentage to the corresponding 'jtid' key in values_2
                values_2[jtid].append(percentage_worked)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_percent_worked, overall_z_score_percent_worked = percent_worked_by_jtid()


# In[7]:


# Test6: uses machinery above to compare AVERAGE SHIFT LENGTHS
def dur_shift_validation():
    """
    Executes a two-sample t-test to compare the average shift length between schedule 1 and schedule 2.
    
    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    dur_shift_ls_1 = []
    for index in range(1, len(visits_df1)+1):
        # Check if the shifts DataFrame is not empty before calculating the average shift length
        if not shifts_df1[index-1].empty:
            dur_shift_ls_1.append(shifts_df1[index-1]['duration'].sum() / len(shifts_df1[index-1])) 
    normality(ls=dur_shift_ls_1)
    # Calculate and display the z score for Schedule 1 based on average shift length
    stability(values1=dur_shift_ls_1, values2=dur_shift_ls_1, \
              title=f'Schedule 1, {z_score(values1=dur_shift_ls_1, values2=dur_shift_ls_1)}')
         
    # Schedule 2
    dur_shift_ls_2 = []
    for index in range(1, len(visits_df2)+1):
        # Check if the shifts DataFrame is not empty before calculating the average shift length
        if not shifts_df2[index-1].empty:
            dur_shift_ls_2.append(shifts_df2[index-1]['duration'].sum() / len(shifts_df2[index-1])) 
    
    normality(ls=dur_shift_ls_2)
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on average shift 
    # length
    stability(values1=dur_shift_ls_1, values2=dur_shift_ls_2, \
              title=f'Schedule 2, {z_score(values1=dur_shift_ls_1, values2=dur_shift_ls_2)}')
    
    # Calculate the p-value using a t-test for comparing the average shift length between schedules
    p_value = t_test(ls1=dur_shift_ls_1, ls2=dur_shift_ls_2)
    
    # Calculate and display the z score for Schedule 2 based on average shift length
    dur_shift_z_score = z_score(values1=dur_shift_ls_1, values2=dur_shift_ls_2)
    stability(values1=dur_shift_ls_1, values2=dur_shift_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, \
              {z_score(values1=dur_shift_ls_1, values2=dur_shift_ls_2)}')
    dur_shift_z_score = z_score(values1=dur_shift_ls_1, values2=dur_shift_ls_2)

    # Plot histograms to compare the distributions of the average shift length for both schedules
    plot_hist(dur_shift_ls_1, dur_shift_ls_2, 'Average shift length', \
              'Shift length', 'Frequency', p_value=p_value)
    
    return p_value, dur_shift_z_score

# Perform the average shift length validation and get the p-value and lists of average shift lengths for 
# both schedules
p_value_dur_shift, dur_shift_z_score = dur_shift_validation()

def dur_shifts_by_jtid():
    """
    Sorts shifts dataframes by JTID and performs t-test on the duration of shifts.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the sum of 'duration' for each 'jtid' as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df1
    for df in shifts_df1:
        # Group by 'jtid' and calculate the sum of 'duration' in each group
        for jtid, duration_sum in df.groupby('jtid')['duration'].sum().items():
            # Append the 'duration' sum to the corresponding 'jtid' key in values_1
            values_1[jtid].append(duration_sum)

    # Schedule 2
    # Initialize a dictionary to store the count of unique 'vid's for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df2
    for df in shifts_df2:
        # Group by 'jtid' and calculate the sum of 'duration' in each group
        for jtid, duration_sum in df.groupby('jtid')['duration'].sum().items():
            # Append the 'duration' sum to the corresponding 'jtid' key in values_2
            values_2[jtid].append(duration_sum)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_dur_shifts, overall_z_score_dur_shifts = dur_shifts_by_jtid()


# In[8]:


# Test7: uses machinery above to compare NUMBER OF HCWs WORKING
def unique_validation():
    """
    Executes a two-sample t-test to compare the number of unique HCWs working during the schedule between
    schedule 1 and schedule 2.
    
    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    unique_ls_1 = []
    for index in range(1, len(visits_df1)+1):
        # Check if the shifts DataFrame is not empty before calculating the number of unique HCWs
        if not shifts_df1[index-1].empty:
            unique_ls_1.append(shifts_df1[index-1]['hid'].nunique())
    normality(ls=unique_ls_1)
    # Calculate and display the z score for Schedule 1 based on the number of unique HCWs
    stability(values1=unique_ls_1, values2=unique_ls_1, \
              title=f'Schedule 1, {z_score(values1=unique_ls_1, values2=unique_ls_1)}')
           
    # Schedule 2
    unique_ls_2 = []
    for index in range(1, len(visits_df2)+1):
        # Check if the shifts DataFrame is not empty before calculating the number of unique HCWs
        if not shifts_df2[index-1].empty:
            unique_ls_2.append(shifts_df2[index-1]['hid'].nunique())
    normality(ls=unique_ls_2)
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on the number of 
    # unique HCWs
    stability(values1=unique_ls_1, values2=unique_ls_2, \
              title=f'Schedule 2, {z_score(values1=unique_ls_2, values2=unique_ls_2)}')
    
    # Calculate the p-value using a t-test for comparing the number of unique HCWs between schedules
    p_value = t_test(ls1=unique_ls_1, ls2=unique_ls_2)
    
    # Calculate and display the z score for Schedule 2 based on the number of unique HCWs
    unique_z_score = z_score(values1=unique_ls_1, values2=unique_ls_2)
    stability(values1=unique_ls_1, values2=unique_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, \
              {z_score(values1=unique_ls_1, values2=unique_ls_2)}')
    unique_z_score = z_score(values1=unique_ls_1, values2=unique_ls_2)

    # Plot histograms to compare the distributions of the number of unique HCWs for both schedules
    plot_hist(unique_ls_1, unique_ls_2, 'Number of unique HCWs working during the schedule', \
              'Number of HCWs', 'Frequency', p_value=p_value)
    
    return p_value, unique_z_score

# Perform the number of unique HCWs validation and get the p-value and lists of the number of unique 
# HCWs for both schedules
p_value_unique, unique_z_score = unique_validation()

def unique_by_jtid():
    """
    Sorts dataframes by JTID and performs t-test on the number of unique HCWs by JTID.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the number of unique HCWs for each 'jtid' as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df1
    for df in shifts_df1:
        # Group by 'jtid' and calculate the number of unique HCWs (hid) in each group
        for jtid, unique_hcws_count in df.groupby('jtid')['hid'].nunique().items():
            # Append the number of unique HCWs to the corresponding 'jtid' key in values_1
            values_1[jtid].append(unique_hcws_count)

    # Schedule 2
    # Initialize a dictionary to store the number of unique HCWs for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df2
    for df in shifts_df2:
        # Group by 'jtid' and calculate the number of unique HCWs (hid) in each group
        for jtid, unique_hcws_count in df.groupby('jtid')['hid'].nunique().items():
            # Append the number of unique HCWs to the corresponding 'jtid' key in values_2
            values_2[jtid].append(unique_hcws_count)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_unique, overall_z_score_unique = unique_by_jtid()


# In[9]:


# Test8: uses machinery above to compare NUMBER OF HCWs WORKING DAY SHIFT
def day_validation():
    """
    Executes a two-sample t-test to compare the average number of HCWs working a day shift between 
    schedule 1 and schedule 2.
    
    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    day_ls_1 = []
    for index in range(1, len(visits_df1)+1):
        # Check if the shifts DataFrame is not empty before filtering and calculating the average number 
        # of HCWs working a day shift
        if not shifts_df1[index-1].empty:
            filtered = shifts_df1[index-1][shifts_df1[index-1]['shift'] == 'day']
            day_ls_1.append(filtered['hid'].nunique())    
    normality(ls=day_ls_1)
    # Calculate and display the z score for Schedule 1 based on the average number of HCWs working 
    # a day shift
    stability(values1=day_ls_1, values2=day_ls_1, \
              title=f'Schedule 1, {z_score(values1=day_ls_1, values2=day_ls_1)}')
        
    # Schedule 2
    day_ls_2 = []
    for index in range(1, len(visits_df2)+1):
        # Check if the shifts DataFrame is not empty before filtering and calculating the average number 
        # of HCWs working a day shift
        if not shifts_df2[index-1].empty:
            filtered = shifts_df2[index-1][shifts_df2[index-1]['shift'] == 'day']
            day_ls_2.append(filtered['hid'].nunique())   
    normality(ls=day_ls_2)
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on the average 
    # number of HCWs working a day shift
    stability(values1=day_ls_1, values2=day_ls_2, \
              title=f'Schedule 2, Similarity Score: {z_score(values1=day_ls_2, values2=day_ls_2)}')
    
    # Calculate the p-value using a t-test for comparing the average number of HCWs working a day shift 
    # between schedules
    p_value = t_test(ls1=day_ls_1, ls2=day_ls_2)
    
    # Calculate and display the z score for Schedule 2 based on the average number of HCWs working 
    # a day shift
    day_z_score = z_score(values1=day_ls_1, values2=day_ls_2)
    stability(values1=day_ls_1, values2=day_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, \
              {z_score(values1=day_ls_1, values2=day_ls_2)}')
    day_z_score = z_score(values1=day_ls_1, values2=day_ls_2)

    # Plot histograms to compare the distributions of the average number of HCWs working a day shift for 
    # both schedules
    plot_hist(day_ls_1, day_ls_2, 'Average number of HCWs working a day shift', \
              'Number of HCWs', 'Frequency', p_value=p_value)

    return p_value, day_z_score

# Perform the average number of HCWs working a day shift validation and get the p-value and lists of the 
# average number of HCWs for both schedules
p_value_day, day_z_score = day_validation()

def day_by_jtid():
    """
    Sorts dataframes by JTID and performs t-test on the number of unique HCWs working the day shift by JTID.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the number of unique HCWs working the day shift for each 'jtid' as 
    # values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df1
    for df in shifts_df1:
        # Filter the DataFrame to include only rows where the 'shifts' column equals 'day'
        df_day_shifts = df.query("shift == 'day'")
        # Group by 'jtid' and calculate the number of unique HCWs working the day shift in each group
        for jtid, unique_hcws_count in df_day_shifts.groupby('jtid')['hid'].nunique().items():
            # Append the number of unique HCWs to the corresponding 'jtid' key in values_1
            values_1[jtid].append(unique_hcws_count)

    # Schedule 2
    # Initialize a dictionary to store the number of unique HCWs working the day shift for each 'jtid' as 
    # values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df2
    for df in shifts_df2:
        # Filter the DataFrame to include only rows where the 'shifts' column equals 'day'
        df_day_shifts = df.query("shift == 'day'")
        # Group by 'jtid' and calculate the number of unique HCWs working the day shift in each group
        for jtid, unique_hcws_count in df_day_shifts.groupby('jtid')['hid'].nunique().items():
            # Append the number of unique HCWs to the corresponding 'jtid' key in values_2
            values_2[jtid].append(unique_hcws_count)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_day, overall_z_score_day = day_by_jtid()


# In[10]:


# Test8: uses machinery above to compare NUMBER OF HCWs WORKING NIGHT SHIFT
def night_validation():
    """
    Executes a two-sample t-test to compare the average number of HCWs working a night shift between 
    schedule 1 and schedule 2.
    
    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    night_ls_1 = []
    for index in range(1, len(visits_df1) + 1):
        # Check if the shifts DataFrame is not empty before filtering and calculating the average number 
        # of HCWs working a night shift
        if not shifts_df1[index - 1].empty:
            filtered = shifts_df1[index - 1][shifts_df1[index - 1]['shift'] == 'night']
            night_ls_1.append(filtered['hid'].nunique())    
    normality(ls=night_ls_1)
    # Calculate and display the z score for Schedule 1 based on the average number of HCWs working 
    # a night shift
    stability(values1=night_ls_1, values2=night_ls_1, \
              title=f'Schedule 1, {z_score(values1=night_ls_1, values2=night_ls_1)}')
        
    # Schedule 2
    night_ls_2 = []
    for index in range(1, len(visits_df2) + 1):
        # Check if the shifts DataFrame is not empty before filtering and calculating the average number 
        # of HCWs working a night shift
        if not shifts_df2[index - 1].empty:
            filtered = shifts_df2[index - 1][shifts_df2[index - 1]['shift'] == 'night']
            night_ls_2.append(filtered['hid'].nunique())   
    normality(ls=night_ls_2)
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on the average 
    # number of HCWs working a night shift
    stability(values1=night_ls_1, values2=night_ls_2, \
              title=f'Schedule 2, Similarity Score: {z_score(values1=night_ls_2, values2=night_ls_2)}')
    
    # Calculate the p-value using a t-test for comparing the average number of HCWs working a night shift 
    # between schedules
    p_value = t_test(ls1=night_ls_1, ls2=night_ls_2)
    
    # Calculate and display the z score for Schedule 2 based on the average number of HCWs working 
    # a night shift
    night_z_score = z_score(values1=night_ls_1, values2=night_ls_2)
    stability(values1=night_ls_1, values2=night_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, \
              {z_score(values1=night_ls_1, values2=night_ls_2)}')
    night_z_score = z_score(values1=night_ls_1, values2=night_ls_2)

    # Plot histograms to compare the distributions of the average number of HCWs working a night shift for 
    # both schedules
    plot_hist(night_ls_1, night_ls_2, 'Average number of HCWs working a night shift', \
              'Number of HCWs', 'Frequency', p_value=p_value)

    return p_value, night_z_score

# Perform the average number of HCWs working a night shift validation and get the p-value and lists of the 
# average number of HCWs for both schedules
p_value_night, night_z_score = night_validation()

def night_by_jtid():
    """
    Sorts dataframes by JTID and performs t-test on the number of unique HCWs working the night shift by JTID.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the number of unique HCWs working the night shift for each 'jtid' as 
    # values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df1
    for df in shifts_df1:
        # Filter the DataFrame to include only rows where the 'shift' column equals 'night'
        df_night_shifts = df.query("shift == 'night'")
        # Group by 'jtid' and calculate the number of unique HCWs working the night shift in each group
        for jtid, unique_hcws_count in df_night_shifts.groupby('jtid')['hid'].nunique().items():
            # Append the number of unique HCWs to the corresponding 'jtid' key in values_1
            values_1[jtid].append(unique_hcws_count)

    # Schedule 2
    # Initialize a dictionary to store the number of unique HCWs working the night shift for each 'jtid' as 
    # values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df2
    for df in shifts_df2:
        # Filter the DataFrame to include only rows where the 'shift' column equals 'night'
        df_night_shifts = df.query("shift == 'night'")
        # Group by 'jtid' and calculate the number of unique HCWs working the night shift in each group
        for jtid, unique_hcws_count in df_night_shifts.groupby('jtid')['hid'].nunique().items():
            # Append the number of unique HCWs to the corresponding 'jtid' key in values_2
            values_2[jtid].append(unique_hcws_count)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_night, overall_z_score_night = night_by_jtid()


# In[11]:


# Test10: uses machinery above to compare NUMBER OF UNIQUE ROOMS VISITED OVER SCHEDULE
def rooms_validation():
    """
    Executes a two-sample t-test to compare the average number of unique rooms visited during the 
    schedule between schedule 1 and schedule 2.
    
    Returns:
        p_value: The p-value resulting from the t-test
    """
    
    # Schedule 1
    rooms_ls_1 = []
    for index in range(1, len(visits_df1)+1):
        # Check if the visits DataFrame is not empty before calculating the average number of unique 
        # rooms visited
        if not visits_df1[index-1].empty:
            rooms_ls_1.append(visits_df1[index-1]['rid'].nunique())     
    normality(ls=rooms_ls_1)
    # Calculate and display the z score for Schedule 1 based on the average number of unique rooms 
    # visited
    stability(values1=rooms_ls_1, values2=rooms_ls_1, \
              title=f'Schedule 1, {z_score(values1=rooms_ls_1, values2=rooms_ls_1)}')
       
    # Schedule 2
    rooms_ls_2 = []
    for index in range(1, len(visits_df2)+1):
        # Check if the visits DataFrame is not empty before calculating the average number of unique 
        # rooms visited
        if not visits_df2[index-1].empty:
            rooms_ls_2.append(visits_df2[index-1]['rid'].nunique())
    normality(ls=rooms_ls_2)
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on the average 
    # number of unique rooms visited
    stability(values1=rooms_ls_1, values2=rooms_ls_2, \
              title=f'Schedule 2, {z_score(values1=rooms_ls_2, values2=rooms_ls_2)}')
    
    # Compare the difference in the means of average number of unique rooms visited
    if statistics.mean(rooms_ls_1) - statistics.mean(rooms_ls_2) < 1:
        # If the difference is less than 1, set p_value to 1 (indicating no significant difference)
        p_value = 1
    else:
        # If the difference is 1 or greater, perform the t-test to get the p-value
        p_value = t_test(ls1=rooms_ls_1, ls2=rooms_ls_2)
    
    # Calculate and display the z score for Schedule 2 based on the average number of unique rooms 
    # visited
    stability(values1=rooms_ls_1, values2=rooms_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, {z_score(values1=rooms_ls_1, values2=rooms_ls_2)}')
    rooms_z_score = z_score(values1=rooms_ls_1, values2=rooms_ls_2)
    
    # Plot histograms to compare the distributions of the average number of unique rooms visited for 
    # both schedules
    plot_hist(rooms_ls_1, rooms_ls_2, 'Average number of unique rooms visited during the schedule', \
              'Number of Rooms', 'Frequency', p_value=p_value)
    
    return p_value, rooms_z_score

# Perform the average number of unique rooms visited validation and get the p-value and lists of the 
# average number of rooms for both schedules
p_value_rooms, rooms_z_score = rooms_validation()

def rooms_by_jtid():
    """
    Sorts dataframes by JTID and performs t-test on the number of unique rooms shift by JTID.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the number of unique rooms for each 'jtid' as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df1
    for df in shifts_df1:
        # Group by 'jtid' and 'hid' and calculate the num of unique rooms visited by each HCW in each group
        for (jtid, hid), unique_rooms_count in df.groupby(['jtid', 'hid'])['rid'].nunique().items():
            # Append the number of unique rooms to the corresponding 'jtid' key in values_1
            values_1[jtid].append(unique_rooms_count)

    # Schedule 2
    # Initialize a dictionary to store the number of unique rooms for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df2
    for df in shifts_df2:
        # Group by 'jtid' and 'hid' and calculate the num of unique rooms visited by each HCW in each group
        for (jtid, hid), unique_rooms_count in df.groupby(['jtid', 'hid'])['rid'].nunique().items():
            # Append the number of unique rooms to the corresponding 'jtid' key in values_2
            values_2[jtid].append(unique_rooms_count)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_rooms, overall_z_score_rooms = rooms_by_jtid()


# In[12]:


# Test11: uses machinery above to compare NUMBER OF UNIQUE ROOMS VISITED PER SHIFT
def rooms_shift_validation():
    """
    Executes a two-sample t-test to compare the average number of unique rooms visited per shift 
    between schedule 1 and schedule 2.
    
    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    rooms_shift_ls_1 = []
    for index in range(1, len(visits_df1)+1):
        # Get the visits and shifts DataFrames for the current schedule
        visits_dataframe = visits_df1[index-1]
        shifts_dataframe = shifts_df1[index-1]

        # Add a new column for shift ID (sid) to both visits and shifts DataFrames
        sids = list(range(len(shifts_dataframe))) 
        shifts_dataframe['sid'] = sids
        visits_dataframe['sid'] = None

        # Match each visit with its corresponding shift based on time and HCW ID
        for _, row in shifts_dataframe.iterrows():
            mask = (visits_dataframe['itime'] <= row['itime']) & (visits_dataframe['otime'] >= \
                    row['otime']) & (visits_dataframe['hid'] == row['hid'])
            matched_visits = visits_dataframe[mask]
            visits_dataframe.loc[mask, 'sid'] = row['sid']

        if not visits_dataframe.empty and not shifts_dataframe.empty:
            # Calculate the average number of unique rooms visited in each shift
            average_unique_rooms_shift = (visits_dataframe.groupby('sid')['rid'].nunique()).mean()
            rooms_shift_ls_1.append(average_unique_rooms_shift)
    normality(ls=rooms_shift_ls_1)
    # Calculate and display the z score for Schedule 1 based on the average number of unique rooms 
    # visited per shift
    stability(values1=rooms_shift_ls_1, values2=rooms_shift_ls_1, \
              title=f'Schedule 1, {z_score(values1=rooms_shift_ls_1, values2=rooms_shift_ls_1)}')
    
    # Schedule 2
    rooms_shift_ls_2 = []
    for index in range(1, len(visits_df2)+1):
        # Get the visits and shifts DataFrames for the current schedule
        visits_dataframe = visits_df2[index-1]
        shifts_dataframe = shifts_df2[index-1]

        # Add a new column for shift ID (sid) to both visits and shifts DataFrames
        sids = list(range(len(shifts_dataframe))) 
        shifts_dataframe['sid'] = sids
        visits_dataframe['sid'] = None

        # Match each visit with its corresponding shift based on time and HCW ID
        for _, row in shifts_dataframe.iterrows():
            mask = (visits_dataframe['itime'] <= row['itime']) & (visits_dataframe['otime'] >= \
                    row['otime']) & (visits_dataframe['hid'] == row['hid'])
            matched_visits = visits_dataframe[mask]
            visits_dataframe.loc[mask, 'sid'] = row['sid']
            
        if not visits_dataframe.empty and not shifts_dataframe.empty:
            # Calculate the average number of unique rooms visited in each shift
            average_unique_rooms_shift = (visits_dataframe.groupby('sid')['rid'].nunique()).mean()
            rooms_shift_ls_2.append(average_unique_rooms_shift)
    normality(ls=rooms_shift_ls_2)
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on the average 
    # number of unique rooms visited per shift
    stability(values1=rooms_shift_ls_2, values2=rooms_shift_ls_2, \
              title=f'Schedule 2, {z_score(values1=rooms_shift_ls_2, values2=rooms_shift_ls_2)}')
    
    # Compare the difference in the means of average unique rooms visited per shift
    if statistics.mean(rooms_shift_ls_1) - statistics.mean(rooms_shift_ls_2) < 1:
        # If the difference is less than 1, set p_value to 1 (indicating no significant difference)
        p_value = 1
    else:
        # If the difference is 1 or greater, perform the t-test to get the p-value
        p_value = t_test(ls1=rooms_shift_ls_1, ls2=rooms_shift_ls_2)
    
    # Calculate and display the z score for Schedule 2 based on the average number of unique rooms 
    # visited per shift
    stability(values1=rooms_shift_ls_1, values2=rooms_shift_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, \
              {z_score(values1=rooms_shift_ls_1, values2=rooms_shift_ls_2)}')
    rooms_shift_z_score = z_score(values1=rooms_shift_ls_1, values2=rooms_shift_ls_2)

    # Plot histograms to compare the distributions of average unique rooms visited per shift for both 
    # schedules
    plot_hist(rooms_shift_ls_1, rooms_shift_ls_2, 'Average number of unique rooms visited per shift', \
              'Number of Rooms', 'Frequency', p_value=p_value)
    
    return p_value, rooms_shift_z_score

# Perform the average number of unique rooms visited per shift validation and get the p-value 
# and lists of average unique rooms visited per shift for both schedules
p_value_rooms_shift, rooms_shift_z_score = rooms_shift_validation()

def rooms_shift_by_jtid():
    """
    Sorts dataframes by JTID and performs t-test on the number of unique rooms per shift by JTID.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the number of unique rooms per shift for each 'jtid' as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df1
    for df in shifts_df1:
        # Group by 'jtid' and 'hid' and calculate the num of unique rooms per shift visited by each HCW in 
        # each group
        for (jtid, hid), unique_rooms_count in df.groupby(['jtid', 'hid'])['rid'].nunique().items():
            # Append the number of unique rooms per shift to the corresponding 'jtid' key in values_1
            values_1[jtid].append(unique_rooms_count)

    # Schedule 2
    # Initialize a dictionary to store the number of unique rooms for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in shifts_df2
    for df in shifts_df2:
        # Group by 'jtid' and 'hid' and calculate the num of unique rooms per shift visited by each HCW in 
        # each group
        for (jtid, hid), unique_rooms_count in df.groupby(['jtid', 'hid'])['rid'].nunique().items():
            # Append the number of unique rooms per shift to the corresponding 'jtid' key in values_2
            values_2[jtid].append(unique_rooms_count)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_rooms_shift, overall_z_score_rooms_shift = rooms_shift_by_jtid()


# In[13]:


# Test12: uses machinery above to compare AVERAGE INTERVAL BETWEEN VISITS BY SAME HCW
def etime_nhitime_validation():
    """
    Executes a two-sample t-test to compare the average time between visit etime and nhitime between
    schedule 1 and schedule 2.
    
    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    etime_nhitime_ls_1 = []
    for index in range(1, len(visits_df1)+1):
        # Convert 'otime' and 'nhitime' columns to pandas datetime objects
        visits_df1[index-1]['otime'] = pd.to_datetime(visits_df1[index-1]['otime'])
        visits_df1[index-1]['nhitime'] = pd.to_datetime(visits_df1[index-1]['nhitime'])
        if not visits_df1[index-1].empty:
            try:
                # Calculate the average time between 'etime' and 'nhitime' in hours
                etime_nhitime_ls_1.append(abs((visits_df1[index-1]['otime'] - \
                                          visits_df1[index-1]['nhitime']).mean().total_seconds() / 3600))
            except:
                # Handle the case where there's an error in the calculation and append None
                etime_nhitime_ls_1.append(None)
    normality(ls=etime_nhitime_ls_1)
    # Calculate and display the z score for Schedule 1 based on the average time between etime and 
    # nhitime
    stability(values1=etime_nhitime_ls_1, values2=etime_nhitime_ls_1, \
              title=f'Schedule 1, {z_score(values1=etime_nhitime_ls_1, values2=etime_nhitime_ls_1)}')
        
    # Schedule 2
    etime_nhitime_ls_2 = []
    for index in range(1, len(visits_df2) + 1):
        # Convert 'otime' and 'nhitime' columns to pandas datetime objects
        visits_df2[index-1]['otime'] = pd.to_datetime(visits_df2[index-1]['otime'])
        visits_df2[index-1]['nhitime'] = pd.to_datetime(visits_df2[index-1]['nhitime'])
        if not visits_df2[index-1].empty:
            try:
                # Calculate the average time between 'etime' and 'nhitime' in hours
                etime_nhitime_ls_2.append(abs((visits_df2[index-1]['otime'] - \
                                          visits_df2[index-1]['nhitime']).mean().total_seconds() / 3600))
            except:
                # Handle the case where there's an error in the calculation and append None
                etime_nhitime_ls_2.append(None)
    normality(ls=etime_nhitime_ls_2)
    # Calculate and display the z score for Schedule 2 based on the average time between etime and 
    # nhitime
    stability(values1=etime_nhitime_ls_2, values2=etime_nhitime_ls_2, \
              title=f'Schedule 2, {z_score(values1=etime_nhitime_ls_2, values2=etime_nhitime_ls_2)}')
    
    # Perform the two-sample t-test to compare the average times between 'etime' and 'nhitime' for both 
    # schedules
    p_value = t_test(ls1=etime_nhitime_ls_1, ls2=etime_nhitime_ls_2)
    
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on the average time 
    # between etime and nhitime
    stability(values1=etime_nhitime_ls_1, values2=etime_nhitime_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, Similarity Score: \
              {z_score(values1=etime_nhitime_ls_1, values2=etime_nhitime_ls_2)}')
    etime_nhitime_z_score = z_score(values1=etime_nhitime_ls_1, values2=etime_nhitime_ls_2)
    
    # Plot histograms to compare the distributions of average times between 'etime' and 'nhitime' for 
    # both schedules
    plot_hist(etime_nhitime_ls_1, etime_nhitime_ls_2, 'Average time between visit etime and nhitime', \
              'Time', 'Frequency', p_value=p_value)
    
    return p_value, etime_nhitime_z_score

# Perform the average time between visit etime and nhitime validation and get the p-value 
# and lists of average times for both schedules
p_value_etime_nhitime, etime_nhitime_z_score = etime_nhitime_validation()

def etime_nhitime_by_jtid():
    """
    Sorts visits dataframes by JTID and performs t-test to measure the time between etime and nhitime.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to measure the time between etime and nhitime for each 'jtid' as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df1
    for df in visits_df1:
        # Convert 'nhitime' and 'otime' columns to datetime objects
        df['nhitime'] = pd.to_datetime(df['nhitime'])
        df['otime'] = pd.to_datetime(df['otime'])

        # Calculate the time difference between 'nhitime' and 'otime' in each group
        df['time_difference'] = (df['nhitime'] - df['otime']).dt.total_seconds() / 3600

        # Calculate the mean time difference per JTID and append to values_1 dictionary
        for jtid, mean_time_diff in df.groupby('jtid')['time_difference'].mean().items():
            values_1[jtid].append(mean_time_diff)

    # Schedule 2
    # Initialize a dictionary to measure the time between otime and nhitime for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df2
    for df in visits_df2:
        # Convert 'nhitime' and 'otime' columns to datetime objects
        df['nhitime'] = pd.to_datetime(df['nhitime'])
        df['otime'] = pd.to_datetime(df['otime'])

        # Calculate the time difference between 'nhitime' and 'etime' in each group
        df['time_difference'] = (df['nhitime'] - df['otime']).dt.total_seconds() / 3600

        # Calculate the mean time difference per JTID and append to values_1 dictionary
        for jtid, mean_time_diff in df.groupby('jtid')['time_difference'].mean().items():
            values_2[jtid].append(mean_time_diff)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_etime_nhitime, overall_z_score_etime_nhitime = etime_nhitime_by_jtid()


# In[14]:


def etime_nritime_validation():
    """
    Executes a two-sample t-test to compare the average time between visit etime and nritime between
    schedule 1 and schedule 2.
    
    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    etime_nritime_ls_1 = []
    for index in range(1, len(visits_df1)+1):
        # Convert 'otime' and 'nritime' columns to pandas datetime objects
        visits_df1[index-1]['otime'] = pd.to_datetime(visits_df1[index-1]['otime'])
        visits_df1[index-1]['nritime'] = pd.to_datetime(visits_df1[index-1]['nritime'])
        if not visits_df1[index-1].empty:
            try:
                # Calculate the average time between 'etime' and 'nritime' in hours
                etime_nritime_ls_1.append(abs((visits_df1[index-1]['otime'] - \
                                          visits_df1[index-1]['nritime']).mean().total_seconds() / 3600))
            except:
                # Handle the case where there's an error in the calculation and append None
                etime_nritime_ls_1.append(None)
    normality(ls=etime_nritime_ls_1)
    # Calculate and display the z score for Schedule 1 based on the average time between etime and 
    # nritime
    stability(values1=etime_nritime_ls_1, values2=etime_nritime_ls_1, \
              title=f'Schedule 1, {z_score(values1=etime_nritime_ls_1, values2=etime_nritime_ls_1)}')
        
    # Schedule 2
    etime_nritime_ls_2 = []
    for index in range(1, len(visits_df2) + 1):
        # Convert 'otime' and 'nritime' columns to pandas datetime objects
        visits_df2[index-1]['otime'] = pd.to_datetime(visits_df2[index-1]['otime'])
        visits_df2[index-1]['nritime'] = pd.to_datetime(visits_df2[index-1]['nritime'])
        if not visits_df2[index-1].empty:
            try:
                # Calculate the average time between 'etime' and 'nritime' in hours
                etime_nritime_ls_2.append(abs((visits_df2[index-1]['otime'] - \
                                          visits_df2[index-1]['nritime']).mean().total_seconds() / 3600))
            except:
                # Handle the case where there's an error in the calculation and append None
                etime_nritime_ls_2.append(None)
    normality(ls=etime_nritime_ls_2)
    # Calculate and display the z score for Schedule 2 based on the average time between etime and 
    # nritime
    stability(values1=etime_nritime_ls_2, values2=etime_nritime_ls_2, \
              title=f'Schedule 2, {z_score(values1=etime_nritime_ls_2, values2=etime_nritime_ls_2)}')
    
    # Perform the two-sample t-test to compare the average times between 'etime' and 'nritime' for both 
    # schedules
    p_value = t_test(ls1=etime_nritime_ls_1, ls2=etime_nritime_ls_2)
    
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on the average time 
    # between etime and nritime
    stability(values1=etime_nritime_ls_1, values2=etime_nritime_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, Similarity Score: \
              {z_score(values1=etime_nritime_ls_1, values2=etime_nritime_ls_2)}')
    etime_nritime_z_score = z_score(values1=etime_nritime_ls_1, values2=etime_nritime_ls_2)
    
    # Plot histograms to compare the distributions of average times between 'etime' and 'nritime' for 
    # both schedules
    plot_hist(etime_nritime_ls_1, etime_nritime_ls_2, 'Average time between visit etime and nritime', \
              'Time', 'Frequency', p_value=p_value)
    
    return p_value, etime_nritime_z_score

# Perform the average time between visit etime and nritime validation and get the p-value 
# and lists of average times for both schedules
p_value_etime_nritime, etime_nritime_z_score = etime_nritime_validation()

def etime_nritime_by_jtid():
    """
    Sorts visits dataframes by JTID and performs t-test to measure the time between etime and nritime.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to measure the time between etime and nritime for each 'jtid' as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df1
    for df in visits_df1:
        # Convert 'nritime' and 'otime' columns to datetime objects
        df['nritime'] = pd.to_datetime(df['nritime'])
        df['otime'] = pd.to_datetime(df['otime'])

        # Calculate the time difference between 'nritime' and 'otime' in each group
        df['time_difference'] = (df['nritime'] - df['otime']).dt.total_seconds() / 3600

        # Calculate the mean time difference per JTID and append to values_1 dictionary
        for jtid, mean_time_diff in df.groupby('jtid')['time_difference'].mean().items():
            values_1[jtid].append(mean_time_diff)

    # Schedule 2
    # Initialize a dictionary to measure the time between otime and nritime for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df2
    for df in visits_df2:
        # Convert 'nritime' and 'otime' columns to datetime objects
        df['nritime'] = pd.to_datetime(df['nritime'])
        df['otime'] = pd.to_datetime(df['otime'])

        # Calculate the time difference between 'nritime' and 'etime' in each group
        df['time_difference'] = (df['nritime'] - df['otime']).dt.total_seconds() / 3600

        # Calculate the mean time difference per JTID and append to values_1 dictionary
        for jtid, mean_time_diff in df.groupby('jtid')['time_difference'].mean().items():
            values_2[jtid].append(mean_time_diff)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_etime_nritime, overall_z_score_etime_nritime = etime_nritime_by_jtid()


# In[15]:


# Test14: uses machinery above to compare PERCENTAGE OF HHYGIENE ON ENTRY
def idisp_validation():
    """
    Executes a two-sample t-test to compare the percentage of idisp between schedule 1 and schedule 2.

    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    idisp_ls_1 = []
    for index in range(1, len(visits_df1) + 1):
        if not visits_df1[index - 1].empty:
            idisp_count = 0
            for inner_index, row in visits_df1[index - 1].iterrows():
                # Count occurrences of 'idisp' with the specified value (e.g., 'rub')
                if row['idisp'] == 'rub':
                    idisp_count += 1
            # Calculate the percentage of occurrences of 'idisp' with the specified value
            idisp_ls_1.append(idisp_count / len(visits_df1[index - 1]))
    normality(ls=idisp_ls_1)
    # Calculate and display the z score for Schedule 1 based on the percentage of 'idisp'
    stability(values1=idisp_ls_1, values2=idisp_ls_1, \
              title=f'Schedule 1, {z_score(values1=idisp_ls_1, values2=idisp_ls_1)}')

    # Schedule 2
    idisp_ls_2 = []
    for index in range(1, len(visits_df2) + 1):
        if not visits_df2[index - 1].empty:
            idisp_count = 0
            for inner_index, row in visits_df2[index - 1].iterrows():
                # Count occurrences of 'idisp' with the specified value (e.g., 'rub')
                if row['idisp'] == 'rub':
                    idisp_count += 1
            # Calculate the percentage of occurrences of 'idisp' with the specified value
            idisp_ls_2.append(idisp_count / len(visits_df2[index - 1]))
    normality(ls=idisp_ls_2)
    # Calculate and display the z score for Schedule 2 based on the percentage of 'idisp'
    stability(values1=idisp_ls_2, values2=idisp_ls_2, \
              title=f'Schedule 2, {z_score(values1=idisp_ls_2, values2=idisp_ls_2)}')
    
    # Perform the two-sample t-test to compare the percentages of 'idisp' for both schedules
    p_value = t_test(ls1=idisp_ls_1, ls2=idisp_ls_2)
    
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on the percentage 
    # of 'idisp'
    stability(values1=idisp_ls_1, values2=idisp_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, {z_score(values1=idisp_ls_1, values2=idisp_ls_2)}')
    idisp_z_score = z_score(values1=idisp_ls_1, values2=idisp_ls_2)

    # Plot histograms to compare the distributions of percentages of 'idisp' for both schedules
    plot_hist(idisp_ls_1, idisp_ls_2, 'Average percentage of idisp', 'Percentage', 'Frequency', \
              p_value=p_value)
    
    return p_value, idisp_z_score

# Perform the idisp validation and get the p-value and lists of percentages for both schedules
p_value_idisp, idisp_z_score = idisp_validation()

def idisp_by_jtid():
    """
    Sorts visits dataframes by JTID and performs t-test on the percentage of idisp with each 'jtid'.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the percentage of idisp for each 'jtid' as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df1
    for df in visits_df1:
        # Group by 'jtid' and calculate the percentage of 'rub' occurrences in the 'idisp' column in each 
        # group
        for jtid, group_df in df.groupby('jtid'):
            total_rows = len(group_df)  # Total number of rows in the group
            rub_count = len(group_df[group_df['idisp'] == 'rub'])  # Count of 'rub' occurrences
            percentage_rub = (rub_count / total_rows) * 100  # Calculate the percentage of 'rub' occurrences
            # Append the percentage of idisp to the corresponding 'jtid' key in values_1
            values_1[jtid].append(percentage_rub)

    # Schedule 2
    # Initialize a dictionary to store the percentage of idisp for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df2
    for df in visits_df2:
        # Group by 'jtid' and calculate the percentage of 'rub' occurrences in the 'idisp' column in each 
        # group
        for jtid, group_df in df.groupby('jtid'):
            total_rows = len(group_df)  # Total number of rows in the group
            rub_count = len(group_df[group_df['idisp'] == 'rub'])  # Count of 'rub' occurrences
            percentage_rub = (rub_count / total_rows) * 100  # Calculate the percentage of 'rub' occurrences
            # Append the percentage of idisp to the corresponding 'jtid' key in values_1
            values_2[jtid].append(percentage_rub)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_idisp, overall_z_score_idisp = visits_by_jtid()


# In[16]:


# Test14: uses machinery above to compare PERCENTAGE OF HHYGIENE ON ENTRY
def odisp_validation():
    """
    Executes a two-sample t-test to compare the percentage of odisp between schedule 1 and schedule 2.

    Returns:
        p_value: The p-value resulting from the t-test
    """
    # Schedule 1
    odisp_ls_1 = []
    for index in range(1, len(visits_df1) + 1):
        if not visits_df1[index - 1].empty:
            odisp_count = 0
            for inner_index, row in visits_df1[index - 1].iterrows():
                # Count occurrences of 'odisp' with the specified value (e.g., 'rub')
                if row['odisp'] == 'rub':
                    odisp_count += 1
            # Calculate the percentage of occurrences of 'odisp' with the specified value
            odisp_ls_1.append(odisp_count / len(visits_df1[index - 1]))
    normality(ls=odisp_ls_1)
    # Calculate and display the z score for Schedule 1 based on the percentage of 'odisp'
    stability(values1=odisp_ls_1, values2=odisp_ls_1, \
              title=f'Schedule 1, {z_score(values1=odisp_ls_1, values2=odisp_ls_1)}')

    # Schedule 2
    odisp_ls_2 = []
    for index in range(1, len(visits_df2) + 1):
        if not visits_df2[index - 1].empty:
            odisp_count = 0
            for inner_index, row in visits_df2[index - 1].iterrows():
                # Count occurrences of 'odisp' with the specified value (e.g., 'rub')
                if row['odisp'] == 'rub':
                    odisp_count += 1
            # Calculate the percentage of occurrences of 'odisp' with the specified value
            odisp_ls_2.append(odisp_count / len(visits_df2[index - 1]))
    normality(ls=odisp_ls_2)
    # Calculate and display the z score for Schedule 2 based on the percentage of 'odisp'
    stability(values1=odisp_ls_2, values2=odisp_ls_2, \
              title=f'Schedule 2, {z_score(values1=odisp_ls_2, values2=odisp_ls_2)}')
    
    # Perform the two-sample t-test to compare the percentages of 'odisp' for both schedules
    p_value = t_test(ls1=odisp_ls_1, ls2=odisp_ls_2)
    
    # Calculate and display the z score between Schedule 1 and Schedule 2 based on the percentage 
    # of 'odisp'
    stability(values1=odisp_ls_1, values2=odisp_ls_2, \
              title=f'Schedule 1 vs. Schedule 2, {z_score(values1=odisp_ls_1, values2=odisp_ls_2)}')
    odisp_z_score = z_score(values1=odisp_ls_1, values2=odisp_ls_2)

    # Plot histograms to compare the distributions of percentages of 'odisp' for both schedules
    plot_hist(odisp_ls_1, odisp_ls_2, 'Average percentage of odisp', 'Percentage', 'Frequency', \
              p_value=p_value)
    
    return p_value, odisp_z_score

# Perform the odisp validation and get the p-value and lists of percentages for both schedules
p_value_odisp, odisp_z_score = odisp_validation()

def odisp_by_jtid():
    """
    Sorts visits dataframes by JTID and performs t-test on the percentage of odisp with each 'jtid'.

    Returns:
        t_tests (Dictionary): Stores p-values of t-tests for each JTID as key of the Dictionary.
    """
    # Schedule 1
    # Initialize a dictionary to store the percentage of odisp for each 'jtid' as values
    values_1 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df1
    for df in visits_df1:
        # Group by 'jtid' and calculate the percentage of 'rub' occurrences in the 'odisp' column in each 
        # group
        for jtid, group_df in df.groupby('jtid'):
            total_rows = len(group_df)  # Total number of rows in the group
            rub_count = len(group_df[group_df['odisp'] == 'rub'])  # Count of 'rub' occurrences
            percentage_rub = (rub_count / total_rows) * 100  # Calculate the percentage of 'rub' occurrences
            # Append the percentage of odisp to the corresponding 'jtid' key in values_1
            values_1[jtid].append(percentage_rub)

    # Schedule 2
    # Initialize a dictionary to store the percentage of odisp for each 'jtid' as values
    values_2 = {key: [] for key in range(1, 33)}
    # Loop through each DataFrame in visits_df2
    for df in visits_df2:
        # Group by 'jtid' and calculate the percentage of 'rub' occurrences in the 'odisp' column in each 
        # group
        for jtid, group_df in df.groupby('jtid'):
            total_rows = len(group_df)  # Total number of rows in the group
            rub_count = len(group_df[group_df['odisp'] == 'rub'])  # Count of 'rub' occurrences
            percentage_rub = (rub_count / total_rows) * 100  # Calculate the percentage of 'rub' occurrences
            # Append the percentage of odisp to the corresponding 'jtid' key in values_1
            values_2[jtid].append(percentage_rub)

    # Run jtid_tests to calculate overall scores for p-value and z-score by JTID
    p_value_score, overall_z_score = jtid_tests(values_1=values_1, values_2=values_2)
    return p_value_score, overall_z_score

p_value_score_odisp, overall_z_score_odisp = odisp_by_jtid()


# # P-Values 

# In[17]:


# Working patterns: averages the p-values for shifts, visits, visits_shift, percent_worked, dur_visits, 
# dur_shift
p_values = [float(p_value_shifts), float(p_value_visits), float(p_value_visits_shift),
            float(p_value_percent_worked), float(p_value_dur_visits), float(p_value_dur_shift)]
print(f'Working patterns: {statistics.mean(p_values)}')
    
# Room scope: averages the p-values for rooms, rooms_shift
p_values = [float(p_value_rooms), float(p_value_rooms_shift)]
print(f'Room scope: {statistics.mean(p_values)}')

# Staffing levels: averages the p-values for unique, day, night
p_values = [float(p_value_unique), float(p_value_day), float(p_value_night)]
print(f'Staffing levels: {statistics.mean(p_values)}')

# Break lengths: averages the p-values for etime_nritime, etime_nhitime
p_values = [float(p_value_etime_nritime), float(p_value_etime_nhitime)]
print(f'Break lengths: {statistics.mean(p_values)}')

# Sanitation: averages the p-values for idisp, odisp
p_values = [float(p_value_idisp), float(p_value_odisp)]
print(f'Sanitation: {statistics.mean(p_values)}')

# P-value Sum
p_values = [float(p_value_shifts), float(p_value_visits), float(p_value_visits_shift),
            float(p_value_percent_worked), float(p_value_dur_visits), float(p_value_dur_shift), 
            float(p_value_rooms), float(p_value_rooms_shift), float(p_value_unique), float(p_value_day), 
            float(p_value_night), float(p_value_etime_nhitime), float(p_value_etime_nritime),
            float(p_value_idisp), float(p_value_odisp)]
print(f'P-values mean: {statistics.mean(p_values)}')

# P-values by JTID
p_values = [float(p_value_score_shifts), float(p_value_score_visits), float(p_value_score_visits_shift),
            float(p_value_score_percent_worked), float(p_value_score_dur_visits), 
            float(p_value_score_dur_shifts), float(p_value_score_rooms), float(p_value_score_rooms_shift), 
            float(p_value_score_unique), float(p_value_score_day), float(p_value_score_night),
            float(p_value_score_etime_nhitime), float(p_value_score_etime_nritime),
            float(p_value_score_idisp), float(p_value_score_odisp)]
print(f'P-values by JTID mean: {np.nanmean(p_values)}')


# # Z-Scores

# In[18]:


# Working patterns: averages the z-scores for shifts, visits, visits_shift, percent_worked, dur_visits, 
# dur_shift
z_scores = [float(z_score_shifts[8:]), float(z_score_visits[8:]), float(z_score_visits_shift[8:]),
            float(percent_worked_z_score[8:]), float(dur_visits_z_score[8:]), 
            float(dur_shift_z_score[8:])]
print(f'Working patterns: {np.nanmean(z_scores)}')
    
# Room scope: averages the z-scores for rooms, rooms_shift
z_scores = [float(rooms_z_score[8:]), float(rooms_shift_z_score[8:])]
print(f'Room scope: {np.nanmean(z_scores)}')

# Staffing levels: averages the z-scores for unique, day, night
z_scores = [float(unique_z_score[8:]), float(day_z_score[8:]), float(night_z_score[8:])]
print(f'Staffing levels: {np.nanmean(z_scores)}')

# Break lengths: averages the z-scores for etime_nritime, etime_nhitime
z_scores = [float(p_value_etime_nritime), float(p_value_etime_nhitime)]
print(f'Break lengths: {np.nanmean(z_scores)}')

# Sanitation: averages the z-scores for idisp, odisp
z_scores = [float(idisp_z_score[8:]), float(odisp_z_score[8:])]
print(f'Sanitation: {np.nanmean(z_scores)}')

# Z-score Sum
z_scores = [float(z_score_shifts[8:]), float(z_score_visits[8:]), float(z_score_visits_shift[8:]),
            float(percent_worked_z_score[8:]), float(dur_visits_z_score[8:]), 
            float(dur_shift_z_score[8:]), float(rooms_z_score[8:]), float(rooms_shift_z_score[8:]), 
            float(unique_z_score[8:]), float(day_z_score[8:]), float(night_z_score[8:]), 
            float(etime_nhitime_z_score[8:]), float(etime_nritime_z_score[8:]),
            float(idisp_z_score[8:]), float(odisp_z_score[8:])]
print(f'Z-scores mean: {np.nanmean(z_scores)}')

# Z-Scores by JTID
z_scores = [float(overall_z_score_shifts), float(overall_z_score_visits), float(overall_z_score_visits_shift),
            float(overall_z_score_percent_worked), float(overall_z_score_dur_visits), 
            float(overall_z_score_dur_shifts), float(overall_z_score_rooms), float(overall_z_score_rooms_shift), 
            float(overall_z_score_unique), float(overall_z_score_day), float(overall_z_score_night),
            float(overall_z_score_etime_nhitime), float(overall_z_score_etime_nritime),
            float(overall_z_score_idisp), float(overall_z_score_odisp)]

print(f'Z-scores by JTID mean: {np.nanmean(z_scores)}')

