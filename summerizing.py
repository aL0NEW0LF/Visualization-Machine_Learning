import pandas as pd

# List of statistical indicators to summarize
stats = ['count', 'mean', 'std', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.50),
         lambda x: x.quantile(0.75)]

# List of file names
files = ['D:/PFE/Statistical_Indicators/test2.xlsx', 'D:/PFE/Statistical_Indicators/test3.xlsx']  # Update with the actual file names


# create an empty list to store the statistical indicators
result_list = []

for file in files:
    # iterate through each file in the folder
    dfs = pd.read_excel(file, sheet_name=None)

    for sheet_name, df in dfs.items():
        print(f"Processing sheet: {sheet_name}")
        file_stats = {'Brearing': sheet_name}
        for col in df.columns:
            # calculate statistical indicators
            max_val = df[col].max()
            min_val = df[col].min()
            mean_val = df[col].mean()
            count_val = df[col].count()
            std_val = df[col].std()
            quantile25_val = df[col].quantile(.25)
            quantile50_val = df[col].quantile(.50)
            quantile75_val = df[col].quantile(.75)
            # add the statistical indicators to the dictionary with column names as numbers
            file_stats[col + '_Count'] = count_val
            file_stats[col + '_Max'] = max_val
            file_stats[col + '_Min'] = min_val
            file_stats[col + '_Mean'] = mean_val
            file_stats[col + '_Std'] = std_val
            file_stats[col + '_25%'] = quantile25_val
            file_stats[col + '_50%'] = quantile50_val
            file_stats[col + '_75%'] = quantile75_val
        # add the sheet statistics to the result list
        result_list.append(file_stats)

    # convert the result list to a dataframe
    result_df = pd.DataFrame(result_list)

# save the result dataframe to an Excel file
result_df.to_excel('D:/PFE/Statistical_Indicators/summary_indicators.xlsx', index=False)
print(f"Statistical indicators saved to summary_indicators.")
