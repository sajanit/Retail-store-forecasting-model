def find_min_max_daterange(df):
    """
    This function takes a DataFrame with a column named 'date' and returns the minimum
    and maximum dates found in that column.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing a 'date' column with datetime objects.

    Returns:
    tuple: A tuple containing two elements:
        - min_date (datetime)
        - max_date (datetime) 
    """
    min_date = df['date'].min()
    max_date = df['date'].max()
#     print(f'date range is from  ,{min_date,max_date}')
    return min_date,max_date
def filter_dataframe(df,date):
    """
    Filter a DataFrame for rows with dates greater than or equal to the given date.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with a 'date' column.
    date (datetime): The date to filter the DataFrame by.

    Returns:
    pandas.DataFrame: The filtered DataFrame.
    """
    
    df_filtered = df.loc[df.date>=date]
    return df_filtered
def promotion_data_preprocessing (promo_data) :
    """
    Fill missing 'onpromotion' values with 0 in the promotion data.

    Parameters:
    promo_data (pandas.DataFrame): The DataFrame containing promotion data.

    Returns:
    pandas.DataFrame: The DataFrame with missing 'onpromotion' values filled.
    """
    
    promo_data.fillna({'onpromotion': 0 }, inplace=True)
#     promo_data['onpromotion'].fillna(0, inplace=True)
    return promo_data
def oil_data_preprocessing(df, oil_df):
    """
    Merge and preprocess oil price data into the main DataFrame.

    This function merges oil price data with the main DataFrame on the 'date' column,
    imputes missing oil prices with the average price for the corresponding year-month,
    and removes temporary columns used for the imputation.

    Parameters:
    df (pandas.DataFrame): The main DataFrame containing the data to be merged.
    oil_df (pandas.DataFrame): The DataFrame containing oil price data.

    Returns:
    pandas.DataFrame: The merged and preprocessed DataFrame.
    """
    df = pd.merge(df, oil_df, how='left', on=['date'])
    # there are missing values in dcoilwtico (oil price).  Let's imputed it using average for the month-year combo
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    oil_df['year'] = oil_df['date'].dt.year
    oil_df['month'] = oil_df['date'].dt.month

    dcoilwtico_mean = oil_df.groupby(['year', 'month'], as_index=False)[['dcoilwtico']].mean()
    dcoilwtico_mean = dcoilwtico_mean.rename(columns = {'dcoilwtico':'dcoilwtico_mean'})
    

    # Replace missing dcoilwtico (oil price) with average dcoilwtico imputed above
    df = pd.merge(df, dcoilwtico_mean, how='left', on=['year', 'month'])
    df.fillna({'dcoilwtico': df['dcoilwtico_mean']}, inplace=True)
#     df['dcoilwtico'].fillna(df['dcoilwtico_mean'], inplace=True)
    df.drop('dcoilwtico_mean', axis=1, inplace=True)
    df.drop(['year', 'month'], axis=1, inplace=True)
    oil_df.drop(['year', 'month'], axis=1, inplace=True)

    df.columns[df.isnull().any()]   # these columns have missing values
    
    return df 
def transaction_data_preprocessing(df,trans_df):
    """
    Preprocess transaction data and merge it with the main DataFrame.

    This function sets the index of the transaction DataFrame to ['store_nbr', 'date'],
    joins it with the main DataFrame, and fills missing transaction values with 0.

    Parameters:
    df (pandas.DataFrame): The main DataFrame to which transaction data will be added.
    trans_df (pandas.DataFrame): The DataFrame containing transaction data.

    Returns:
    pandas.DataFrame: The merged DataFrame with transaction data included.
    """
    trans_df = trans_df.set_index(["store_nbr", "date"])[["transactions"]]
    # Join the two DataFrames
    df = df.join(trans_df, how='left')
#     df.columns[df.isnull().any()]   # these columns have missing values
#     df['transactions'].fillna(0, inplace=True)
    df.fillna({'transactions': 0}, inplace=True)
    return df
def create_date_features(df):
    """
    Extract date-related features from the 'date' column in the DataFrame.

    - 'day_of_week': Day of the week (0=Monday, 6=Sunday).
    - 'month': Month of the year.
    - 'day': Day of the month.
    - 'week': ISO week number.
    - 'year': Year.
    - 'is_weekend': Boolean indicating if the date is a weekend.
    - 'Dayofyear': Day of the year.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with a 'date' column.

    Returns:
    pandas.DataFrame: The DataFrame with new date-related features.
    """
        #Extract date-related features

    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    df['Dayofyear'] = df['date'].dt.dayofyear
        
    return df
def create_holiday_features (df,hol_df):
    """
    Add holiday features to the DataFrame based on the holiday information.

    This function merges holiday data into the main DataFrame, categorizes holidays
    by their locale (city, state, national), and imputes missing values for holiday types 
    and descriptions.

    Parameters:
    df (pandas.DataFrame): The main DataFrame to which holiday data will be added.
    hol_df (pandas.DataFrame): The DataFrame containing holiday information.

    Returns:
    pandas.DataFrame: The DataFrame with holiday features added.
    """
    

    hol_df.drop(hol_df[hol_df.transferred == True].index, inplace=True)
        # Break holidays into separate dataframes
    holidays_cities = hol_df[hol_df.locale == "Local"]       # city level holidays
    holidays_states = hol_df[hol_df.locale == "Regional"]    # state level holidays
    holidays_national = hol_df[hol_df.locale == "National"]  # national holidays

        # # Rename columns to help with joining dataframes later
    holidays_cities = holidays_cities.rename(columns = {'locale_name':'city', 'type':'holiday_type'})
    holidays_states = holidays_states.rename(columns = {'locale_name':'state', 'type':'holiday_type'})
    holidays_national = holidays_national.rename(columns = {'type':'holiday_type'})
        # # We don't need locale_name at all for national holidays
    holidays_national.drop('locale_name', axis=1, inplace=True)
        
        # # locale column is useless - let's drop it to simplify joining dataframes
    holidays_cities.drop('locale', axis=1, inplace=True)
    holidays_states.drop('locale', axis=1, inplace=True)
    holidays_national.drop('locale', axis=1, inplace=True)

        # # transferred column is now useless - let's drop it to simplify joining dataframes
    holidays_cities.drop('transferred', axis=1, inplace=True)
    holidays_states.drop('transferred', axis=1, inplace=True)
    holidays_national.drop('transferred', axis=1, inplace=True)
        
        
    df = pd.merge(df, holidays_cities, how='left', on=['date', 'city'])
#         df.columns[df.isnull().any()]   # these columns have missing values
    df = df.rename(columns = {'holiday_type':'holiday_type_city', 'description':'description_city'})


    df = pd.merge(df, holidays_states, how='left', on=['date', 'state'])
#         df.columns[df.isnull().any()]   # these columns have missing values
    df = df.rename(columns = {'holiday_type':'holiday_type_state', 'description':'description_state'})

    df = pd.merge(df, holidays_national, how='left', on=['date'])
#         df.columns[df.isnull().any()]   # these columns have missing values
    df.rename(columns = {'holiday_type':'holiday_type_nat', 'description':'description_nat'}, inplace=True)

        # Impute missing values
    df.fillna({'holiday_type_city': 'no holiday'}, inplace=True)
    df.fillna({'holiday_type_state': 'no holiday'}, inplace=True)
    df.fillna({'holiday_type_nat': 'no holiday'}, inplace=True)
    df.fillna({'description_city': 'no holiday'}, inplace=True)
    df.fillna({'description_state': 'no holiday'}, inplace=True)
    df.fillna({'description_nat': 'no holiday'}, inplace=True)
    

#         df.columns[df.isnull().any()]   # these columns have missing values
        
    return df
def create_paydate_feature(df):
    """
    Add a 'PayDay' feature to the DataFrame based on mid-month and end-of-month paydays.

    This function identifies the 15th day and the last day of each month as paydays and adds
    a binary 'PayDay' column to the DataFrame, where 1 indicates a payday and 0 otherwise.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with 'date', 'day', and 'Dayofyear' columns.

    Returns:
    pandas.DataFrame: The DataFrame with the 'PayDay' feature added.
    """
        
        # Get mid-month paydays
    mid_month = df['Dayofyear'][df['day'] == 15].unique()

        # Get end-of-month paydays
    end_month = df['Dayofyear'][df['date'].dt.is_month_end].unique()

        # Combine mid-month and end-of-month paydays
    paydates = np.append(mid_month, end_month)
    paydates

        #Adding pay days data  
    df['PayDay'] = np.where(np.isin(df.Dayofyear, paydates), 1, 0)

    df.Dayofyear[df.PayDay == 1].unique()

    return df
def custom_label_encoding(df):
    """
    Perform custom label encoding for categorical columns in the DataFrame.

    This function encodes specific categorical columns in the DataFrame using custom dictionaries.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with categorical columns to be encoded.

    Returns:
    pandas.DataFrame: The DataFrame with encoded categorical columns.
    """
        
    data = ['Quito', 'Guayaquil', 'Cuenca', 'Santo Domingo', 'Ambato', 'Machala', 'Manta', 'Latacunga',
             'Loja', 'Daule', 'Cayambe', 'Babahoyo', 'Esmeraldas', 'Libertad', 'Salinas', 'Ibarra',
             'Quevedo', 'Guaranda', 'Puyo', 'Riobamba', 'El Carmen', 'Playas']

        # Create a dictionary to map each city to its index
    mapping = {a: i for i, a in enumerate(data)}
    df['city'] = df['city'].map(mapping)


    data = ['Pichincha', 'Guayas', 'Azuay', 'Santo Domingo de los Tsachilas', 'Tungurahua', 'Manabi', 'El Oro',
                         'Los Rios', 'Cotopaxi', 'Loja', 'Esmeraldas', 'Santa Elena', 'Imbabura', 'Bolivar', 'Pastaza', 'Chimborazo']

        # Create a dictionary to map each city to its index
    mapping = {a: i for i, a in enumerate(data)}
    df['state'] = df['state'].map(mapping)

    data = ['A','B','C','D','E']

    # Create a dictionary to map each city to its index
    mapping = {a: i for i, a in enumerate(data)}
    df['type'] = df['type'].map(mapping)


    data = ['GROCERY I', 'BEVERAGES', 'CLEANING', 'PRODUCE', 'DAIRY', 'PERSONAL CARE', 'BREAD/BAKERY',
                         'HOME CARE', 'DELI', 'MEATS', 'POULTRY', 'FROZEN FOODS', 'EGGS', 'LIQUOR,WINE,BEER', 
                         'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'PREPARED FOODS', 'CELEBRATION', 
                         'PLAYERS AND ELECTRONICS', 'LADIESWEAR', 'LAWN AND GARDEN', 'AUTOMOTIVE', 'PET SUPPLIES', 
                         'GROCERY II', 'LINGERIE', 'SEAFOOD', 'BEAUTY', 'MAGAZINES', 'SCHOOL AND OFFICE SUPPLIES', 
                         'HARDWARE', 'HOME APPLIANCES', 'BABY CARE', 'BOOKS']

        # Create a dictionary to map each city to its index
    mapping = {a: i for i, a in enumerate(data)}
    df['family'] = df['family'].map(mapping)

    data = ['no holiday', 'Holiday', 'Transfer', 'Additional', 'Event']
        # Create a dictionary to map each city to its index
    mapping = {a: i for i, a in enumerate(data)}
    df['holiday_type_city'] = df['holiday_type_city'].map(mapping)
    df['holiday_type_state'] = df['holiday_type_state'].map(mapping)
    df['holiday_type_nat'] = df['holiday_type_nat'].map(mapping)

    data = ['no holiday', 'Fundacion de Guayaquil-1',
                'Fundacion de Guayaquil',
                'Fundacion de Cuenca', 
                'Fundacion de Santo Domingo',
                'Fundacion de Machala',
                'Cantonizacion de Latacunga',
                        'Cantonizacion de Cayambe',
                'Fundacion de Manta', 'Cantonizacion de Libertad','Fundacion de Esmeraldas',
                'Cantonizacion de Riobamba',
                        'Fundacion de Riobamba',
                'Cantonizacion de Guaranda', 'Cantonizacion del Puyo', 
                'Cantonizacion de El Carmen',
                'Provincializacion de Cotopaxi',
                        'Provincializacion de Imbabura', 'Carnaval', 'Dia del Trabajo', 'Dia de la Madre-1',
                        'Traslado Primer dia del ano', 'Traslado Batalla de Pichincha', 'Dia de la Madre',
                        'Traslado Primer Grito de Independencia','Viernes Santo']

        # Create a dictionary to map each city to its index
    mapping = {a: i for i, a in enumerate(data)}
    df['description_city'] = df['description_city'].map(mapping)
    df['description_state'] = df['description_state'].map(mapping)
    df['description_nat'] = df['description_nat'].map(mapping)
        
        
    return df
def filter_dataframe_training_validation(df,start,end):
    """
    Filter the DataFrame for a specified date range for training and validation.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing date information.
    start (datetime): The start date of the range (inclusive).
    end (datetime): The end date of the range (inclusive).

    Returns:
    pandas.DataFrame: The filtered DataFrame containing data within the specified date range.
    """

# Assuming your DataFrame is named df
    filtered_df = df[(df['date'] >= start) & (df['date'] <= end)]
    return filtered_df
def find_feature_importance(train_df):
    """
    This function calculates feature importance using a trained model, creates a DataFrame
    for visualization, and plots the feature importances in a horizontal bar plot.

    Parameters:
    train_df (pandas.DataFrame): The DataFrame containing training data features.

    Returns:
    None
    """
    
    # Calculate feature importance
    feature_importances = model.get_feature_importance()
    feature_names = train_df.columns

    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()
def calculate_score(pred, act, weight, chunk_size=1000):
    """
    Calculate the Normalized Weighted Root Mean Squared Logarithmic Error (NWRMSLE).

    Parameters:
    pred (array-like): Predicted values.
    act (array-like): Actual values.
    weight (array-like): Weight for each data point.
    chunk_size (int): Size of the chunks for processing data.

    Returns:
    float: The calculated NWRMSLE score.
    """
    # Convert pred and act to numpy arrays
    pred =  (np.array(pred)).T
    act = np.array(act)
    per_weight = np.array(weight)
    # Calculate the number of chunks
    num_chunks = int(np.ceil(len(pred) / chunk_size))
    
    # Initialize total nwrmsle
    total_nwrmsle = 0
    
    # Process data in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(pred))
        
        # Get chunk of data
        chunk_pred = pred[start_idx:end_idx]
        chunk_act = act[start_idx:end_idx]
        
        # Calculate nwrmsle for chunk
        chunk_nwrmsle = np.sum(per_weight[start_idx:end_idx] * np.square(chunk_pred - chunk_act))
        
        # Add chunk nwrmsle to total
        total_nwrmsle += chunk_nwrmsle
    
    # Calculate final nwrmsle
    nwrmsle = math.sqrt(total_nwrmsle / np.sum(per_weight))
    
    return nwrmsle
def find_min_max_daterange(df):
    """
    This function takes a DataFrame with a column named 'date' and returns the minimum
    and maximum dates found in that column.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing a 'date' column with datetime objects.

    Returns:
    tuple: A tuple containing two elements:
        - min_date (datetime)
        - max_date (datetime) 
    """
    min_date = df['date'].min()
    max_date = df['date'].max()
#     print(f'date range is from  ,{min_date,max_date}')
    return min_date,max_date
def filter_dataframe(df,date):
    """
    Filter a DataFrame for rows with dates greater than or equal to the given date.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with a 'date' column.
    date (datetime): The date to filter the DataFrame by.

    Returns:
    pandas.DataFrame: The filtered DataFrame.
    """
    
    df_filtered = df.loc[df.date>=date]
    return df_filtered
def promotion_data_preprocessing (promo_data) :
    """
    Fill missing 'onpromotion' values with 0 in the promotion data.

    Parameters:
    promo_data (pandas.DataFrame): The DataFrame containing promotion data.

    Returns:
    pandas.DataFrame: The DataFrame with missing 'onpromotion' values filled.
    """
    
    promo_data.fillna({'onpromotion': 0 }, inplace=True)
#     promo_data['onpromotion'].fillna(0, inplace=True)
    return promo_data
def oil_data_preprocessing(df, oil_df):
    """
    Merge and preprocess oil price data into the main DataFrame.

    This function merges oil price data with the main DataFrame on the 'date' column,
    imputes missing oil prices with the average price for the corresponding year-month,
    and removes temporary columns used for the imputation.

    Parameters:
    df (pandas.DataFrame): The main DataFrame containing the data to be merged.
    oil_df (pandas.DataFrame): The DataFrame containing oil price data.

    Returns:
    pandas.DataFrame: The merged and preprocessed DataFrame.
    """
    df = pd.merge(df, oil_df, how='left', on=['date'])
    # there are missing values in dcoilwtico (oil price).  Let's imputed it using average for the month-year combo
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    oil_df['year'] = oil_df['date'].dt.year
    oil_df['month'] = oil_df['date'].dt.month

    dcoilwtico_mean = oil_df.groupby(['year', 'month'], as_index=False)[['dcoilwtico']].mean()
    dcoilwtico_mean = dcoilwtico_mean.rename(columns = {'dcoilwtico':'dcoilwtico_mean'})
    

    # Replace missing dcoilwtico (oil price) with average dcoilwtico imputed above
    df = pd.merge(df, dcoilwtico_mean, how='left', on=['year', 'month'])
    df.fillna({'dcoilwtico': df['dcoilwtico_mean']}, inplace=True)
#     df['dcoilwtico'].fillna(df['dcoilwtico_mean'], inplace=True)
    df.drop('dcoilwtico_mean', axis=1, inplace=True)
    df.drop(['year', 'month'], axis=1, inplace=True)
    oil_df.drop(['year', 'month'], axis=1, inplace=True)

    df.columns[df.isnull().any()]   # these columns have missing values
    
    return df 
def transaction_data_preprocessing(df,trans_df):
    """
    Preprocess transaction data and merge it with the main DataFrame.

    This function sets the index of the transaction DataFrame to ['store_nbr', 'date'],
    joins it with the main DataFrame, and fills missing transaction values with 0.

    Parameters:
    df (pandas.DataFrame): The main DataFrame to which transaction data will be added.
    trans_df (pandas.DataFrame): The DataFrame containing transaction data.

    Returns:
    pandas.DataFrame: The merged DataFrame with transaction data included.
    """
    trans_df = trans_df.set_index(["store_nbr", "date"])[["transactions"]]
    # Join the two DataFrames
    df = df.join(trans_df, how='left')
#     df.columns[df.isnull().any()]   # these columns have missing values
#     df['transactions'].fillna(0, inplace=True)
    df.fillna({'transactions': 0}, inplace=True)
    return df
def create_date_features(df):
    """
    Extract date-related features from the 'date' column in the DataFrame.

    - 'day_of_week': Day of the week (0=Monday, 6=Sunday).
    - 'month': Month of the year.
    - 'day': Day of the month.
    - 'week': ISO week number.
    - 'year': Year.
    - 'is_weekend': Boolean indicating if the date is a weekend.
    - 'Dayofyear': Day of the year.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with a 'date' column.

    Returns:
    pandas.DataFrame: The DataFrame with new date-related features.
    """
        #Extract date-related features

    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    df['Dayofyear'] = df['date'].dt.dayofyear
        
    return df
def create_holiday_features (df,hol_df):
    """
    Add holiday features to the DataFrame based on the holiday information.

    This function merges holiday data into the main DataFrame, categorizes holidays
    by their locale (city, state, national), and imputes missing values for holiday types 
    and descriptions.

    Parameters:
    df (pandas.DataFrame): The main DataFrame to which holiday data will be added.
    hol_df (pandas.DataFrame): The DataFrame containing holiday information.

    Returns:
    pandas.DataFrame: The DataFrame with holiday features added.
    """
    

    hol_df.drop(hol_df[hol_df.transferred == True].index, inplace=True)
        # Break holidays into separate dataframes
    holidays_cities = hol_df[hol_df.locale == "Local"]       # city level holidays
    holidays_states = hol_df[hol_df.locale == "Regional"]    # state level holidays
    holidays_national = hol_df[hol_df.locale == "National"]  # national holidays

        # # Rename columns to help with joining dataframes later
    holidays_cities = holidays_cities.rename(columns = {'locale_name':'city', 'type':'holiday_type'})
    holidays_states = holidays_states.rename(columns = {'locale_name':'state', 'type':'holiday_type'})
    holidays_national = holidays_national.rename(columns = {'type':'holiday_type'})
        # # We don't need locale_name at all for national holidays
    holidays_national.drop('locale_name', axis=1, inplace=True)
        
        # # locale column is useless - let's drop it to simplify joining dataframes
    holidays_cities.drop('locale', axis=1, inplace=True)
    holidays_states.drop('locale', axis=1, inplace=True)
    holidays_national.drop('locale', axis=1, inplace=True)

        # # transferred column is now useless - let's drop it to simplify joining dataframes
    holidays_cities.drop('transferred', axis=1, inplace=True)
    holidays_states.drop('transferred', axis=1, inplace=True)
    holidays_national.drop('transferred', axis=1, inplace=True)
        
        
    df = pd.merge(df, holidays_cities, how='left', on=['date', 'city'])
#         df.columns[df.isnull().any()]   # these columns have missing values
    df = df.rename(columns = {'holiday_type':'holiday_type_city', 'description':'description_city'})


    df = pd.merge(df, holidays_states, how='left', on=['date', 'state'])
#         df.columns[df.isnull().any()]   # these columns have missing values
    df = df.rename(columns = {'holiday_type':'holiday_type_state', 'description':'description_state'})

    df = pd.merge(df, holidays_national, how='left', on=['date'])
#         df.columns[df.isnull().any()]   # these columns have missing values
    df.rename(columns = {'holiday_type':'holiday_type_nat', 'description':'description_nat'}, inplace=True)

        # Impute missing values
    df.fillna({'holiday_type_city': 'no holiday'}, inplace=True)
    df.fillna({'holiday_type_state': 'no holiday'}, inplace=True)
    df.fillna({'holiday_type_nat': 'no holiday'}, inplace=True)
    df.fillna({'description_city': 'no holiday'}, inplace=True)
    df.fillna({'description_state': 'no holiday'}, inplace=True)
    df.fillna({'description_nat': 'no holiday'}, inplace=True)
    

#         df.columns[df.isnull().any()]   # these columns have missing values
        
    return df
def create_paydate_feature(df):
    """
    Add a 'PayDay' feature to the DataFrame based on mid-month and end-of-month paydays.

    This function identifies the 15th day and the last day of each month as paydays and adds
    a binary 'PayDay' column to the DataFrame, where 1 indicates a payday and 0 otherwise.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with 'date', 'day', and 'Dayofyear' columns.

    Returns:
    pandas.DataFrame: The DataFrame with the 'PayDay' feature added.
    """
        
        # Get mid-month paydays
    mid_month = df['Dayofyear'][df['day'] == 15].unique()

        # Get end-of-month paydays
    end_month = df['Dayofyear'][df['date'].dt.is_month_end].unique()

        # Combine mid-month and end-of-month paydays
    paydates = np.append(mid_month, end_month)
    paydates

        #Adding pay days data  
    df['PayDay'] = np.where(np.isin(df.Dayofyear, paydates), 1, 0)

    df.Dayofyear[df.PayDay == 1].unique()

    return df
def custom_label_encoding(df):
    """
    Perform custom label encoding for categorical columns in the DataFrame.

    This function encodes specific categorical columns in the DataFrame using custom dictionaries.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with categorical columns to be encoded.

    Returns:
    pandas.DataFrame: The DataFrame with encoded categorical columns.
    """
        
    data = ['Quito', 'Guayaquil', 'Cuenca', 'Santo Domingo', 'Ambato', 'Machala', 'Manta', 'Latacunga',
             'Loja', 'Daule', 'Cayambe', 'Babahoyo', 'Esmeraldas', 'Libertad', 'Salinas', 'Ibarra',
             'Quevedo', 'Guaranda', 'Puyo', 'Riobamba', 'El Carmen', 'Playas']

        # Create a dictionary to map each city to its index
    mapping = {a: i for i, a in enumerate(data)}
    df['city'] = df['city'].map(mapping)


    data = ['Pichincha', 'Guayas', 'Azuay', 'Santo Domingo de los Tsachilas', 'Tungurahua', 'Manabi', 'El Oro',
                         'Los Rios', 'Cotopaxi', 'Loja', 'Esmeraldas', 'Santa Elena', 'Imbabura', 'Bolivar', 'Pastaza', 'Chimborazo']

        # Create a dictionary to map each city to its index
    mapping = {a: i for i, a in enumerate(data)}
    df['state'] = df['state'].map(mapping)

    data = ['A','B','C','D','E']

    # Create a dictionary to map each city to its index
    mapping = {a: i for i, a in enumerate(data)}
    df['type'] = df['type'].map(mapping)


    data = ['GROCERY I', 'BEVERAGES', 'CLEANING', 'PRODUCE', 'DAIRY', 'PERSONAL CARE', 'BREAD/BAKERY',
                         'HOME CARE', 'DELI', 'MEATS', 'POULTRY', 'FROZEN FOODS', 'EGGS', 'LIQUOR,WINE,BEER', 
                         'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'PREPARED FOODS', 'CELEBRATION', 
                         'PLAYERS AND ELECTRONICS', 'LADIESWEAR', 'LAWN AND GARDEN', 'AUTOMOTIVE', 'PET SUPPLIES', 
                         'GROCERY II', 'LINGERIE', 'SEAFOOD', 'BEAUTY', 'MAGAZINES', 'SCHOOL AND OFFICE SUPPLIES', 
                         'HARDWARE', 'HOME APPLIANCES', 'BABY CARE', 'BOOKS']

        # Create a dictionary to map each city to its index
    mapping = {a: i for i, a in enumerate(data)}
    df['family'] = df['family'].map(mapping)

    data = ['no holiday', 'Holiday', 'Transfer', 'Additional', 'Event']
        # Create a dictionary to map each city to its index
    mapping = {a: i for i, a in enumerate(data)}
    df['holiday_type_city'] = df['holiday_type_city'].map(mapping)
    df['holiday_type_state'] = df['holiday_type_state'].map(mapping)
    df['holiday_type_nat'] = df['holiday_type_nat'].map(mapping)

    data = ['no holiday', 'Fundacion de Guayaquil-1',
                'Fundacion de Guayaquil',
                'Fundacion de Cuenca', 
                'Fundacion de Santo Domingo',
                'Fundacion de Machala',
                'Cantonizacion de Latacunga',
                        'Cantonizacion de Cayambe',
                'Fundacion de Manta', 'Cantonizacion de Libertad','Fundacion de Esmeraldas',
                'Cantonizacion de Riobamba',
                        'Fundacion de Riobamba',
                'Cantonizacion de Guaranda', 'Cantonizacion del Puyo', 
                'Cantonizacion de El Carmen',
                'Provincializacion de Cotopaxi',
                        'Provincializacion de Imbabura', 'Carnaval', 'Dia del Trabajo', 'Dia de la Madre-1',
                        'Traslado Primer dia del ano', 'Traslado Batalla de Pichincha', 'Dia de la Madre',
                        'Traslado Primer Grito de Independencia','Viernes Santo']

        # Create a dictionary to map each city to its index
    mapping = {a: i for i, a in enumerate(data)}
    df['description_city'] = df['description_city'].map(mapping)
    df['description_state'] = df['description_state'].map(mapping)
    df['description_nat'] = df['description_nat'].map(mapping)
        
        
    return df
def filter_dataframe_training_validation(df,start,end):
    """
    Filter the DataFrame for a specified date range for training and validation.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing date information.
    start (datetime): The start date of the range (inclusive).
    end (datetime): The end date of the range (inclusive).

    Returns:
    pandas.DataFrame: The filtered DataFrame containing data within the specified date range.
    """

# Assuming your DataFrame is named df
    filtered_df = df[(df['date'] >= start) & (df['date'] <= end)]
    return filtered_df
def find_feature_importance(train_df):
    """
    This function calculates feature importance using a trained model, creates a DataFrame
    for visualization, and plots the feature importances in a horizontal bar plot.

    Parameters:
    train_df (pandas.DataFrame): The DataFrame containing training data features.

    Returns:
    None
    """
    
    # Calculate feature importance
    feature_importances = model.get_feature_importance()
    feature_names = train_df.columns

    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()
def calculate_score(pred, act, weight, chunk_size=1000):
    """
    Calculate the Normalized Weighted Root Mean Squared Logarithmic Error (NWRMSLE).

    Parameters:
    pred (array-like): Predicted values.
    act (array-like): Actual values.
    weight (array-like): Weight for each data point.
    chunk_size (int): Size of the chunks for processing data.

    Returns:
    float: The calculated NWRMSLE score.
    """
    # Convert pred and act to numpy arrays
    pred =  (np.array(pred)).T
    act = np.array(act)
    per_weight = np.array(weight)
    # Calculate the number of chunks
    num_chunks = int(np.ceil(len(pred) / chunk_size))
    
    # Initialize total nwrmsle
    total_nwrmsle = 0
    
    # Process data in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(pred))
        
        # Get chunk of data
        chunk_pred = pred[start_idx:end_idx]
        chunk_act = act[start_idx:end_idx]
        
        # Calculate nwrmsle for chunk
        chunk_nwrmsle = np.sum(per_weight[start_idx:end_idx] * np.square(chunk_pred - chunk_act))
        
        # Add chunk nwrmsle to total
        total_nwrmsle += chunk_nwrmsle
    
    # Calculate final nwrmsle
    nwrmsle = math.sqrt(total_nwrmsle / np.sum(per_weight))
    
    return nwrmsle
