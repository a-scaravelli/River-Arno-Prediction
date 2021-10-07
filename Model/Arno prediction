from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


class arno_prediction:
    def __init__(self):
        pass

    def upload_df(self, path='', df=''):
        '''
        upload a csv file if a path is provided
        '''
        if path == '':
            self.df = df
        else:
            self.df = pd.read_csv(path)
            print('df loaded succesfully')
            print(self.df.describe)

    def get_df(self, old='no'):
        '''
        insert the df into a variable
        '''
        if old == 'si':
            return self.df_old
        else:
            return self.df

    def print_df(self):
        print(self.df)

    def is_date_date_time(self, date='Date'):
        '''
        check if the date is already in format Date time
        '''
        if self.df[date].dtype == '<M8[ns]':
            pass
        else:
            self.df[date] = pd.to_datetime(self.df[date], format='%d/%m/%Y')

    def create_time_columns(self, datecolumn):
        '''
        Create year and month column based on datecolumn
        '''
        self.is_date_date_time(datecolumn)

        self.df['year'] = [d.year for d in self.df[datecolumn]]
        self.df['month'] = [d.strftime('%m') for d in self.df[datecolumn]]

    def null_count(self):
        '''
        returns the columns with null values and how many per each column
        '''
        null_counts = self.df.isnull().sum() / self.df.shape[0]
        null_counts[null_counts > 0].sort_values(ascending=False)
        print(null_counts)
        return null_counts

    def draw_plot_trend(self, x, y, title='', xlabel='year', ylabel='Hydrometry_Nave_di_Rosano', dpi=100):
        '''
        plot the yearly trend of a column
        in:
        x - the column that you want to use as the x axis
        y - the column that you want to use as the y axis

        out:
        graph
        '''
        self.is_date_date_time()

        plt.figure(figsize=(16, 5), dpi=dpi)
        plt.plot(self.df[x], self.df[y], color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()

    def draw_yearly_monthly_box_plot(self, df, y):
        '''
        plot two boxplot,
        one with years and on with months

        in:
        df - dataframe that you want to use (needs to have a Date column)
        y - column you want to use

        out:
        two graphs
        '''
        self.create_time_columns('Date')

        fig, axes = plt.subplots(2, 1, figsize=(35, 20), dpi=80)
        sns.boxplot(x='year', y=y, data=df, ax=axes[0])
        sns.boxplot(x='month', y=y, data=df.loc[~df['year'].isin([1998, 2020]), :])

        # Set Title
        axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18);
        axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
        plt.show()

    def corr_mat(self):
        '''
        plots the correlation matrix of self.df
        '''
        correlation_mat = self.df.corr()
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(correlation_mat, annot=True)
        plt.show()

    def fill_column(self, value, date, df_new_to_fill, column_to_fill):
        '''
        if value is null,
        it gets the value of the column "column_to_fill" from the df_new_to_fill df
        where there is a match in date.

        in:
        value - value you want to check is null
        date - the date you want to retrieve the value from
        df_new_to_fill - df from which you want to get the value
        column_to_fill - column of the df you want to get the value from

        output:
        value - value you want to retrieve


        '''
        if pd.isnull(value):
            value = list(df_new_to_fill[df_new_to_fill['Date'] == date][column_to_fill])[0]
        return value

    def fill_null(self, df):
        '''
        fill the null of a df using Random forest.
        starts with a df with only full columns and the column with the least amount of nulls
        uses a random forest to get predict the nulls.

        repeats the process untill there are no more null values

        in:
        df - df you want to fill, the df

        out:
        df - df with one column filled

        '''

        # creo la serie con ogni colonna ed il numero di nulli, ordinato in ordine decrescente
        null_counts = df.isnull().sum() / df.shape[0]
        null_counts_ordered = null_counts.sort_values(ascending=False)

        # creo un df con solo le colonne senza nulli, piu la colonna con meno nulli
        columns = list(null_counts_ordered[null_counts_ordered == 0].index)
        column_to_fill = null_counts_ordered[null_counts_ordered > 0].index[-1]
        columns.append(column_to_fill)
        df_new = df[columns].copy()

        # creo due dataset, uno dove ho la colonna da riempire valorizza, ed uno dove e' nulla
        df_new_full = df_new[df_new[column_to_fill].notnull()].copy()
        df_new_to_fill = df_new[df_new[column_to_fill].isnull()].copy()

        # creo X e y, e relativo train e test
        X = df_new_full.drop(['Date', column_to_fill, 'Hydrometry_Nave_di_Rosano'], axis=1).copy()
        # X_date = df_new_full['Date'].copy()
        y = df_new_full[column_to_fill].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # inizializzo il regressore, fitto e traino
        regr = RandomForestRegressor()
        regr.fit(X_train, y_train)

        y_pred = regr.predict(X_test)

        rmse = sqrt(mean_squared_error(y_test, y_pred))
        print(column_to_fill, ':', rmse)

        # se l-rmse e' troppo basso, droppo la colonna, se invece mi va bene sostituisco i valori
        if rmse < 10:
            X = df_new_to_fill.drop(['Date', column_to_fill, 'Hydrometry_Nave_di_Rosano'], axis=1).copy()
            df_new_to_fill[column_to_fill] = regr.predict(X)

            df[column_to_fill] = df.apply(
                lambda row: self.fill_column(row[column_to_fill], row['Date'], df_new_to_fill, column_to_fill), axis=1)

            return df

        else:
            df = df.drop([column_to_fill], axis=1).copy()

            return df

    def fill_temp(self, temp, month):
        '''

        checks if temp is null, if it is, it retrieves a new temp value from df_temperature using the month

        in:
        temp: integer
        month : str

        out: integer
        '''
        if pd.isnull(temp):
            temp = self.df_temperature[month]
        return temp

    def fill_na(self):
        '''
        delets all rows before date '01-01-2004', because they are all empty
        drops all the rows where the target variable is null

        fills 'temperature_firenze' based on the month

        fills al the remaining colums

        out:
        self.df_nona - df filled
        '''

        df_cleaned = self.df[self.df['Date'] >= '01-01-2004'].copy()
        df_cleaned = df_cleaned.dropna(subset=['Hydrometry_Nave_di_Rosano'])

        # riempire temparatura firenze

        self.df_temperature = df_cleaned.groupby('month')['Temperature_Firenze'].mean()

        df_cleaned['Temperature_Firenze'] = df_cleaned.apply(
            lambda row: self.fill_temp(row['Temperature_Firenze'], row['month']), axis=1)

        null_counts = df_cleaned.isnull().sum() / df_cleaned.shape[0]
        columns_to_fill = len(null_counts[null_counts > 0].sort_values(ascending=False))

        for column in list(range(columns_to_fill)):
            print(column)
            df_cleaned = self.fill_null(df_cleaned)

        self.df_old = self.df
        self.df_nona = df_cleaned
        return self.df_nona

    def shift_x_days(self, df, number_of_days, column):
        '''
        creates n lags columns

        in:
        df - df you want to use
        number_of_days - integer of how many columns you want
        column - column you want to create the lag for

        out:
        df - df with lag columns
        '''
        for i in list(range(number_of_days)):
            day = i + 1
            df[column + '_t' + str(day)] = df[column].shift(day)
        return df

    def shift_1_year(self, df, string_to_search):
        '''
        creates a lag of 1 year for columns with a certain pattern

        in:
        df - df you want to use
        string_to_search - pattern you want to search inside the column names

        '''
        for column in list(df.columns):
            if (string_to_search in column) and ('_t' not in column):
                df[column + '_1year'] = df[column].shift(365)
        return df

    def preprocessing(self, df, predict='no'):
        '''
        creates a map for the target variable, and creates a new column with the avg target variable based on the month
        shifts the target variable by -1, so that the data of yesterday can be used to predict tomorrow
        creates lags

        in:
        df - dataframe you want to preprocess
        predict - str used for skipping some steps

        out:
        self.df_preprocessed - the preprocessed df

        '''
        if predict == 'no':
            self.df_arno = pd.DataFrame(df.groupby('month')['Hydrometry_Nave_di_Rosano'].mean()).reset_index()
            self.df_arno.columns = ['month', 'monthly_avg']
            df['Arno_Domani'] = df['Hydrometry_Nave_di_Rosano'].shift(-1)
            df = df.drop(['Hydrometry_Nave_di_Rosano'], axis=1)

        df = df.merge(self.df_arno, how='left', on='month')
        df = self.shift_x_days(df, 10, 'Arno_Domani')

        for column in list(df.columns):
            if ('Rainfall' in column):
                df = self.shift_x_days(df, 5, column)
                """df[column+'_t+1'] = df[column].shift(-1)  
                df[column+'_prevision'] = df.apply(lambda row: 1 if row[column+'_t+1']>0 else 0,axis = 1)
                df = df.drop([column+'_t+1'],axis = 1)"""
            if ('Temperature_Firenze' in column):
                df = self.shift_x_days(df, 5, column)
        if predict == 'no':
            self.df_preprocessed = df.dropna()
        return self.df_preprocessed

    def train_model(self, df_ready):
        '''
        creates X and y
        creates a Pipeline with StandarScaler and Random Forest
        uses gridsearch to search some params

        train the model
        fit the model
        predict on test df

        calculate rms

        in:
        df_ready - df you want to train and predict on

        out:
        model
        '''
        X = df_ready.drop(['Date', 'Arno_Domani'], axis=1).copy()
        # X_date = df_new_full['Date'].copy()
        y = df_ready['Arno_Domani'].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # inizializzo il regressore, fitto e traino

        regr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regr', RandomForestRegressor())
        ])

        # grid_params = {'regr__n_estimators': (50,100),
        #             'regr__max_depth': (5,10,15)
        # }

        grid_params = {'regr__n_estimators': (50, 100)
                       }

        clf = GridSearchCV(regr_pipeline, grid_params)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        rmse = sqrt(mean_squared_error(y_test, y_pred))

        self.model = clf
        return clf
        print(rmse)

    def predict(self, df, df_ready):
        '''
        produce prediction based on the self.model

        in:
        df - df with new rows that you want to predict
        df_ready - df with old data, already processed



        '''

        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['year'] = [d.year for d in df['Date']]
        df['month'] = [d.strftime('%m') for d in df['Date']]

        for index, row in df.iterrows():
            df_temp = df.iloc[[index]]
            columns = list(df_temp.columns) + ['Arno_Domani']
            df_for_prediction = df_ready[columns].tail(20)

            df2 = pd.concat([df_for_prediction, df_temp])

            df2 = self.preprocessing(df2, predict='yes')
            # return df
            # df = df.drop(['Date','Arno_Domani'],axis = 1).tail(1)
            df2 = df2.tail(1)
            date_temp = df2['Date']
            df2 = df2.drop(['Date', 'Arno_Domani'], axis=1)

            prediction = self.model.predict(df2)

            df2['Arno_Domani'] = prediction
            df2.insert(0, "Date", [date_temp])

            # print('The prediction for the date ',date_temp,' is ',prediction)
            print('The prediction is ', prediction)

            df_ready = pd.concat([df_ready, df2])




arno = arno_prediction()
arno.upload_df(path = '../Datasets/River_Arno.csv')
arno.create_time_columns('Date')

arno.draw_plot_trend('Date','Hydrometry_Nave_di_Rosano')

df = arno.get_df()
arno.draw_yearly_monthly_box_plot(df,'Hydrometry_Nave_di_Rosano')

df_nona = arno.fill_na()

df = arno.get_df()
arno.draw_yearly_monthly_box_plot(df,'Rainfall_Incisa')

df_old = arno.get_df(old = 'si')
arno.draw_yearly_monthly_box_plot(df_old,'Rainfall_Incisa')

df_ready = arno.preprocessing(df_nona)

model = arno.train_model(df_ready)

df_test = arno.get_df()
df_test = df_test.tail(1).drop(['Hydrometry_Nave_di_Rosano','year','month'],axis = 1)

df_test_2 = df_test.tail(2)[:]

df_test_2['Date'] = '2020-07-01'

df_test = pd.concat([df_test,df_test_2]).reset_index().drop(['index'],axis = 1)

df_test.at[0,'Date'] = '30/06/2020'
df_test.at[1,'Date'] = '01/07/2020'

df_test

arno.predict(df_test,df_ready)

model.best_estimator_.named_steps["regr"].feature_importances_