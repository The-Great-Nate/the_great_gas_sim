'''
This code was adapted from SCIF20002 assessment 01. The class has been repurposed to serve gas_sim.cpp
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import os
from IPython.display import display, HTML
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from typing import Union, Tuple, Any


class Particles:
    """
    This class holds a database positions, velocities, accelerations, kinetic and potential energies of particles..
    """
    def __init__(self, database_name:str) -> None:
        """
        Creates a private attribute of the database imported from a file.
        :param database_name (string) : The name the database file in the filesystem.
        :param database_name (pd.DataFrame) : The DataFrame so methods of this class can be used.
        Attributes
        ----------
        __database : Pandas Data Frame
        __parameters : Dictionary of system parameters
        __N, __dt, __steps, __box_size, __duration : Attributes storing each respective system parameter. Prevents running extract_parameters() & casts parameters to correct datatype.
        """
        self.file_path = f"data\{database_name}"
        self.__database = pd.read_csv(self.file_path, delimiter="\t", skiprows = 6, index_col=False)
        self.__parameters = self.extract_parameters()
        self.__database = self.__database[:-1]
        self.__N = int(self.__parameters["N"])
        self.__dt = float(self.__parameters["dt"])
        self.__steps = int(self.__parameters["steps"])
        self.__box_size = float(self.__parameters["box_size"])
        self.__duration = str(self.__parameters["duration"])

    def get_df(self) -> pd.DataFrame:
        """
        Returns the value of __database.
        :return: dataframe in __database
        """
        return self.__database
    
    def get_N(self) -> int:
        """
        Returns the value of __N.
        :return: N
        """
        return self.__N
    
    def get_dt(self) -> float:
        """
        Returns the value of __dt.
        :return: dataframe in __dt
        """
        return self.__dt
    
    def get_steps(self) -> int:
        """
        Returns the value of __steps.
        :return: dataframe in __steps
        """
        return self.__steps
    
    def get_box_size(self) -> float:
        """
        Returns the value of __box_size.
        :return: dataframe in __box_size
        """
        return self.__box_size
    
    def get_duration(self) -> str:
        """
        Returns the value of __duration.
        :return: dataframe in __duration
        """
        return self.__duration
    
    def set_df(self, df: pd.DataFrame) -> None:
        """
        Sets __database to the parameter
        :param df: database to replace the current instance of __database
        :return: None
        """
        self.__database = df

    def extract_parameters(self) -> dict:
        """
        Extracts system parameters from data file.
        :return: dict of parameters
        """
        parameters = {}
        file = open(self.file_path, "r")
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line == "table":
                break
            params = line.split("=")
            parameters[params[0]] = params[1]
        #to get duration
        line = lines[-1]
        params = line.split("=")
        parameters[params[0]] = params[1]
        return parameters

    def get_speed(self, data: pd.DataFrame = None) -> pd.Series:
        """
        Returns pandas series of particle's speed from performing pythagoras on each component.
        :param data: if none, use __database. Otherwise use data passed in.
        :return: pandas series of speeds
        """
        if data == None:
            speeds = (self.__database["vx"]**2 + self.__database["vy"]**2 + self.__database["vz"]**2)**(1/2)
            return speeds
        else:
            speeds = (data["vx"]**2 + data["vy"]**2 + data["vz"]**2)**(1/2)
            return speeds

    def get_total_kinetic_per_step(self) -> pd.DataFrame:
        """
        Returns pandas DataFrame of the total kinetic energy of the system during each timestep.
        :return: pandas DataFrame of time and total kinetic energy
        """
        temp_data = self.__database.copy()
        temp_data["speed"] = self.get_speed()
        time_steps = np.arange(0, self.__steps * self.__dt, self.__dt)
        KE_s = np.zeros(self.__steps)
        temp_data["KE"] = 0.5 * (temp_data["speed"]**2)
        for time, group in temp_data.groupby("t"):
            total_ke_time = np.sum(group["KE"])
            KE_s[int(float(time)/self.__dt)] = total_ke_time
        KEt_df = pd.DataFrame({"t":time_steps, "Total Kinetic Energy per Timestep":KE_s})
        return KEt_df
    
    def get_total_potential_per_step(self) -> pd.DataFrame:
        """
        Returns pandas DataFrame of the total potential energy of the system during each timestep.
        :return: pandas DataFrame of time and total potential energy
        """
        temp_data = self.__database.copy()
        time_steps = np.arange(0, self.__steps * self.__dt, self.__dt)
        PE_s = np.zeros(self.__steps)
        for time, group in temp_data.groupby("t"):
            total_pe_time = np.sum(group["PE"])
            PE_s[int(float(time)/self.__dt)] = total_pe_time
        PEt_df = pd.DataFrame({"t":time_steps, "Total Potential Energy per Timestep":PE_s})
        return PEt_df
    
    def get_total_momentum_per_step(self) -> pd.DataFrame:
        """
        Returns pandas DataFrame of the total momentum when collisions with the wall occur during each timestep.
        :return: pandas DataFrame of time and total momentum when collisions with the wall occur
        """
        temp_data = self.__database.copy()
        time_steps = np.arange(0, self.__steps * self.__dt, self.__dt)
        MOMx = np.zeros(self.__steps)
        MOMy = np.zeros(self.__steps)
        MOMz = np.zeros(self.__steps)
        for time, group in temp_data.groupby("t"):
            total_MOMx_time = np.sum(group["MOMx"])
            total_MOMy_time = np.sum(group["MOMy"])
            total_MOMz_time = np.sum(group["MOMz"])
            MOMx[int(float(time)/self.__dt)] = total_MOMx_time
            MOMy[int(float(time)/self.__dt)] = total_MOMy_time
            MOMz[int(float(time)/self.__dt)] = total_MOMz_time
        MOM_df = pd.DataFrame({"t":time_steps, 
                               "Total MOMx":MOMx,
                               "Total MOMy":MOMy,
                               "Total MOMz":MOMz,})
        return MOM_df
    
    def get_pressure_exerted_per_step(self, data = None) -> pd.DataFrame:
        """
        Returns pandas DataFrame of the pressure when collisions with the wall occur during each timestep.
        :param: data, can be none, or if input, the pressure will be calculated from the input if there are MOMx, MOMy, MOMz Columns.
        :return: pandas DataFrame of time and pressure when collisions with the wall occur
        """
        if data is None:
            total_momentums_time = self.get_total_momentum_per_step()
            Px = total_momentums_time["Total MOMx"] / (self.__box_size**2 * self.__dt)
            Py = total_momentums_time["Total MOMy"] / (self.__box_size**2 * self.__dt)
            Pz = total_momentums_time["Total MOMz"] / (self.__box_size**2 * self.__dt)
            P_df = pd.DataFrame({"t":total_momentums_time["t"], 
                                "Total Px":Px,
                                "Total Py":Py,
                                "Total Pz":Pz,})
            return P_df
        else:
            total_momentums_time = data
            dt = data["t"][1] - data["t"][0]
            Px = total_momentums_time["Total MOMx"] / (self.__box_size**2 * dt)
            Py = total_momentums_time["Total MOMy"] / (self.__box_size**2 * dt)
            Pz = total_momentums_time["Total MOMz"] / (self.__box_size**2 * dt)
            P_df = pd.DataFrame({"t":total_momentums_time["t"], 
                                "Total Px":Px,
                                "Total Py":Py,
                                "Total Pz":Pz,})
            return P_df
    
    def resample_by_mean(self, time_step:str, df:pd.DataFrame = None) -> pd.DataFrame:
        """
        Returns pandas DataFrame of resampled data by timeframe user inputs
        :param: data, can be none, or if input, the Input will be resampled data by timeframe user inputs if it has valid time column
        :return: pandas DataFrame resampled data
        """
        if df is None:
            resampled_data = self.__database.copy()
            if "n" not in list(resampled_data.columns):
                return
            else:
                resampled_df = pd.DataFrame({})
                for n, group in resampled_data.groupby("n"):
                    group["t"] = pd.to_numeric(group["t"], errors="coerce")
                    group.index = pd.to_timedelta(group["t"], unit="s")
                    group = group.resample(time_step).mean()
                    resampled_df = pd.concat([resampled_df, group], ignore_index=True)
                resampled_df = resampled_df.sort_values("t")
                resampled_df.reset_index(inplace=True, drop = True)
                return resampled_df
        else:
            resampled_data = df
            if "n" not in list(resampled_data.columns):
                resampled_data["t"] = pd.to_numeric(resampled_data["t"], errors="coerce")
                resampled_data.index = pd.to_timedelta(resampled_data["t"], unit="s")
                resampled_data = resampled_data.resample(time_step).mean()
                return resampled_data
            else:
                resampled_df = pd.DataFrame({})
                for n, group in resampled_data.groupby("n"):
                    group["t"] = pd.to_numeric(group["t"], errors="coerce")
                    group.index = pd.to_timedelta(group["t"], unit="s")
                    group = group.resample(time_step).mean()
                    resampled_df = pd.concat([resampled_df, group], ignore_index=True)
                resampled_df = resampled_df.sort_values("t")
                resampled_df.reset_index(inplace=True, drop = True)
                return resampled_df
            

    def get_percent_errs_over_duration(self, data, time:pd.Series = None, col = None, plot:bool = False, save:bool = False) -> np.array:
        """
        Returns array of durations and percentage errors
        :return: durations, duration of data taken
        :return: percent_errs, percentage errors wrt duration
        """
        if isinstance(data, str):
            if data not in list(self.__database.columns):
                durations = np.arange(0, self.__dt * self.__steps, self.__dt)
                percent_errs = np.zeros(self.__steps)
                for i in range(self.__steps):
                    if i == 0:
                        pass
                    else:
                        mean_no_resample = self.get_mean(self.__database[:i][data])
                        std_err_no_resample = self.get_standard_error(self.__database[:i][data])
                        percent_err = np.abs(std_err_no_resample / mean_no_resample)
                        percent_errs[i] = percent_err

                    if plot == True:
                        plt.figure(figsize=(10,6))
                        plt.plot(durations, percent_errs)
                        plt.xlabel("duration (s)")
                        plt.ylabel("Percentage Error")
                        plt.title("Percentage Error against The Duration the Simulation was Ran")
                        if save == True:
                            plt.savefig(f"duration_plot_{self.file_path[6:]}_{data}.pdf", format="pdf")
                        plt.show()
                return durations, percent_errs
        if isinstance(data, pd.DataFrame):
            if time is None:
                if col is None:
                    raise TypeError("get_percent_errs_over_duration() must have parameter for \"col\" if inputting data frame.")
                
                time_diffs = np.diff(data["t"])
                dt = np.mean(time_diffs)
                steps = len(data)
                durations = np.arange(0, dt * steps, dt)
                percent_errs = np.zeros(steps)
                for i in range(steps):
                    if i == 0:
                        pass
                    else:
                        mean_no_resample = self.get_mean(data[:i][col])
                        std_err_no_resample = self.get_standard_error(data[:i][col])
                        percent_err = np.abs(std_err_no_resample / mean_no_resample)
                        percent_errs[i] = percent_err

                if plot == True:
                    plt.figure(figsize=(10,6))
                    plt.plot(durations, percent_errs)
                    plt.xlabel("duration (s)")
                    plt.ylabel("Percentage Error")
                    plt.title("Percentage Error against The Duration the Simulation was Ran")
                    if save == True:
                        if dt != self.__dt:
                            plt.savefig(f"duration_plot_{self.file_path[6:]}_{col}_RESAMPLED_{dt}.pdf", format="pdf")
                        else:
                            plt.savefig(f"duration_plot_{self.file_path[6:]}_{col}.pdf", format="pdf")
                    plt.show()
                return durations, percent_errs

    def get_file_size(self) -> float:
        """
        Returns file path
        """
        return os.path.getsize(self.file_path)

    def concat(self, dfs:list[pd.DataFrame], join:str = "Outer", axis:int = 1):
        """
        Joins 2 data frams together
        :return: concatenated data frame
        """
        return pd.concat(dfs, join = join, axis = axis)
    

    def display_data_info(self) -> None:
        """
        Output Number of Fields, Records and Column Names
        :return: None
        """
        print(f"Number of fields: {len(self.__database.columns)}")
        print(f"Number of records: {len(self.__database)}")
        print(f"Column Names: {list(self.__database.columns)}")

    def return_col_names(self) -> [str]:
        """
        Returns column names as list in __database
        :return: [str]
        """
        return list(self.__database.columns)

    def check_missing_values(self) -> pd.Series:
        """
        Returns number of NaN values in each column
        :return: pandas series
        """
        return self.__database.isnull().sum()

    def return_complete_records(self) -> pd.DataFrame:
        """
        Returns NaN-less records (rows)
        :return: Records with no NaN values
        """
        return self.__database.dropna()

    def summary(self, df=None) -> pd.DataFrame:
        """
        Returns summary statistics of dataframe
        :param df: Optional, returns summary statistics of data frame input.
         Otherwise, if none, summary of __database is output.
        :return: summary statistics of df or __database if df = None
        """
        if df is None:
            df = self.__database
        else:
            pass
        return df.describe()

    def corr(self) -> pd.DataFrame:
        """
        Returns correlation summary between variable pairs as dataframe.
        :return: Correlation summary between variable pairs
        """
        return self.__database.corr(numeric_only=True)

    def heatmap(self, square=True, vmin=-1, vmax=1, cmap="RdBu") -> None:
        """
        Outputs heatmap based on correlation summary data frame
        :param square: Default:True.  Axes aspect to “equal” so each cell will be square-shaped.
        :param vmin: Default:-1. Maximum negative value to anchor to colormap.
        :param vmax: Default:1. Maximum positive value to anchor to colormap.
        :param cmap: Default:"R dBu". Colourmap that data would map to.
        :return: None
        """
        sns.heatmap(data=self.corr(), square=square, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.title("Correlation Heatmap")
        plt.show()

    def return_particle_data(self, name: int) -> Union[None, pd.DataFrame]:
        """
        Returns record of particle index is assigned to
        :param name: particle index to return record for.
        :return: None if index is not in database | record for particle.
        """
        if name > np.max(self.__database["n"]):
            print("This is not a particle in the database")
            return
        else:
            return self.__database[self.__database["n"] == name]

    def return_corr(self, col_1: str, col_2: str) -> Union[None, int]:
        """
        Returns pearsons r value between two variables
        :param col_1: column 1
        :param col_2: column 2
        :return: None if either parameter doesn't have a valid column name | pearsons r
        """
        if col_1 not in list(self.__database.columns) or col_2 not in list(self.__database.columns):
            print("Enter a valid column name in the database")
            return
        else:
            return self.__database[col_1].corr(self.__database[col_2])

    def plot_relationship(self, col_1: str, col_2: str, hue=None) -> None:
        """
        Returns None if neither parameter is a valid column name.
        Outputs either:
        • countplot (quantity of observations in each categorical bin)
        • catplot (Set to plot box plot to show distribution between categorical var and numeric)
        • rel (scatterplot between col_1 and col_2)
        :param col_1: column 1
        :param col_2: column 2
        :param hue: Grouping variable that will produce elements with different colors. Ideally use a categorical variable.
        :return: None
        """
        if col_1 not in list(self.__database.columns) or col_2 not in list(self.__database.columns):
            print("Enter a valid column name in the database")
            return

        '''
        filter_nan is necessary. 
        The check for categorical variables below tries to convert values in a column to a
        numeric format. If any value cant be converted to numeric, it is NaN. A boolean mask is applied and if entire
        column is False, the column is numeric. However if column has a NaN value before this is even done, the column
        is assumed to be categorical which is bad. 
        '''
        filter_nan = self.__database.dropna(subset=[col_1, col_2])
        if pd.to_numeric(filter_nan[col_1], errors="coerce").notna().all() == False and pd.to_numeric(filter_nan[col_2],
                                                                                                      errors="coerce").notna().all() == False:
            print("Comparing 2 categorical variables")
            sns.countplot(data=self.__database, x=col_1, hue=col_2)
        elif pd.to_numeric(filter_nan[col_1], errors="coerce").notna().all() == False or pd.to_numeric(
                filter_nan[col_2], errors="coerce").notna().all() == False:
            sns.catplot(data=self.__database, x=col_1, y=col_2, kind="box", aspect=1.5, hue=hue)
        else:
            fig, ax = plt.subplots(figsize = (10,6))
            self.__database.plot(ax = ax, x=col_1, y=col_2)
            ax.set_title(f"\"{col_1}\" x \"{col_2}\"", fontsize=16)

    def pairwise_plots(self, hue=None) -> None:
        """
        Outputs pairwise relationship plots in the dataset
        :param hue: Grouping variable that will produce elements in each plot with different colors.
         Ideally use a categorical variable.
        :return: None
        """
        sns.pairplot(self.__database, hue=hue)

    def filter_for_characteristic(self, column: str, characteristic: str) -> pd.DataFrame:
        """
        Returns record(s) that are under the characteristic within the column specified
        :param column: Column name to check characteristic against
        :param characteristic: Characteristic checked for in each record.
        :return: DataFrame of records that satisfy characteristic
        """
        if column not in list(self.__database.columns):
            print("This is not a column in the database")
            return
        else:
            # use boolean indexing to filter rows.
            return self.__database[self.__database[column] == characteristic]

    def return_unique_values(self, col: str) -> list:
        """
        Utilises the DataFrame.unique() method to output all unique values within a column specified
        :param col: Column to check for unique values
        :return: List of unique values in col
        """
        return self.__database[col].unique()

    '''
    Static methods are used to boost program performance. The @staticmethod decorator can be used on certain methods
    as some methods can be used independently no matter the instance of the class.
    '''


    def get_model(self, col_1: str, col_2: str, fit_intercept=True) -> tuple:
        """
        Creates instance of LinearRegression that accepts training data from both columns and then fits. The training data
        created is through sklearn.model_selection's test_train_split() method. Due to
        Returns tuple of:
        • model: Object of LinearRegression
        • x_test: testing dataset to use in later prediction.
        • y_test: testing dataset to check reliability of model
        • col_1: column 1 (X)
        • col_2: column 2 (Y/target)
        :param col_1: Training Data
        :param col_2: Target Values
        :param fit_intercept: Default: True, assumes there is or is not an intercept to take into account
        :return: Tuple stated above
        """
        model = linear_model.LinearRegression(fit_intercept=fit_intercept)
        '''
        There is only 1 feature in the X data so the data takes the shape (samples, features = 1)
        The target values only need 1 dimension.
        '''
        x = self.__database[[col_1]]
        y = self.__database[col_2]

        '''
        To prevent overfitting and enable the model to create new data in any scanario rather than only relying
        on the entire database only, the data is split into a training set and test set, where the test set is
        30% and by default, the training size is set to "compliment the test size"
        '''
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        # Fitting the model to the training data created from train_test_split()
        model.fit(x_train, y_train)
        return model, x_test, y_test, col_1, col_2

    @staticmethod
    def get_model_parameters(model: tuple) -> Union[str, tuple]:
        """
        Returns tuple of either warning message (String) or (int)
        :param model: Tuple generated from self.get_model(). In order,
        • model: Object of LinearRegression
        • x_test: testing dataset to use in later prediction.
        • y_test: testing dataset to check reliability of model
        • col_1: column 1 (X)
        • col_2: column 2 (Y/target)
        :return: warning message or tuple of gradient and y intercept
        """
        if not isinstance(model[0], linear_model.LinearRegression):
            return "This is not a LinearRegression object"
        model_obj = model[0]
        return model_obj.coef_[0], model_obj.intercept_

    @staticmethod
    def predict(model: tuple) -> Union[str, tuple]:
        """
        Returns prediction from inputted model tuple generated from get_model()
        :param model: Tuple generated from self.get_model(). In order,
        • model: Object of LinearRegression
        • x_test: testing dataset to use in later prediction.
        • y_test: testing dataset to check reliability of model
        • col_1: column 1 (X)
        • col_2: column 2 (Y/target)
        :return: Warning message | array of predicted values from model.
        """
        if not isinstance(model[0], linear_model.LinearRegression):
            return "This is not a LinearRegression object"
        pred = model[0].predict(model[1])
        return pred

    @staticmethod
    def plot_prediction(model: tuple, pred: [int]) -> Union[str, None]:
        """
        Returns either warning message or plots model results and observed values against X set as a scatter and line
        plot respectively.
        :param model: Tuple generated from self.get_model(). In order,
        • model: Object of LinearRegression
        • x_test: testing dataset to use in later prediction.
        • y_test: testing dataset to check reliability of model
        • col_1: column 1 (X)
        • col_2: column 2 (Y/target)
        :param pred: Prediction generated from predict()
        :return: Warning Message | None
        """
        if not isinstance(model[0], linear_model.LinearRegression):
            return "This is not a LinearRegression object"
        reg_model = model[0]
        data_for_predict = model[1]
        y_col = model[2]
        plt.scatter(data_for_predict, y_col, label=f"{model[3]} test data", marker="x")
        plt.plot(data_for_predict, pred, label=f"Prediction", color="orange")
        # Add axis labels and title
        plt.title(f"{model[3]} & prediction x {model[3]}")
        plt.xlabel(model[3])
        plt.ylabel(model[4])
        plt.legend()

    @staticmethod
    def plot_residuals(model: tuple, pred: [int]) -> None:
        """
        Plots residuals between observed target values and predicted target values against X test data.
        This is calculated through subtracting the observed target values by the predicted.
        :param model: Tuple generated from self.get_model(). In order,
        • model: Object of LinearRegression
        • x_test: testing dataset to use in later prediction.
        • y_test: testing dataset to check reliability of model
        • col_1: column 1 (X)
        • col_2: column 2 (Y/target)
        :param pred: Prediction generated from predict()
        :return: None
        """
        plt.plot(model[1], model[2] - pred, '.')
        # Add a horizontal line at zero to guide the eye
        plt.axhline(0, color='k', linestyle='dashed')
        # Add axis labels
        plt.xlabel(model[3])
        plt.ylabel("Residuals")

    @staticmethod
    def output_model_worth(model: tuple, pred: [int]) -> tuple[float | Any, float | Any]:
        """
        Returns R^2 and Root Mean Squared Error. Methods imported from sklearn library
        :param model: Tuple generated from self.get_model(). In order,
        • model: Object of LinearRegression
        • x_test: testing dataset to use in later prediction.
        • y_test: testing dataset to check reliability of model
        • col_1: column 1 (X)
        • col_2: column 2 (Y/target)
        :param pred: Prediction generated from predict()
        :return: Tuple with R^2 value and root mean squared speed.
        """
        r2 = r2_score(model[2], pred)
        rmse = mean_squared_error(model[2], pred, squared=False)
        print(f"r2_score: {r2}")
        print(f"root mean squared error: {rmse}")
        return r2, rmse
    
    def get_mean(self, vals:pd.Series = None):
        if vals is None:
            raise TypeError("get_mean() missing 1 required positional argument: \"vals\"")
        else:    
            mean = np.mean(vals)
            return mean

    def get_standard_error(self, vals:pd.Series = None):
        if vals is None:
            raise TypeError("get_standard_error() missing 1 required positional argument: \"vals\"")
        else:    
            standard_errors = np.std(vals) / np.sqrt(len(vals))
            return standard_errors
        

    def make_animation_2d(self, axis1: str , axis2: str, save: bool = True) -> animation:
        """
        WARNING: PLEASE MAKE SURE FFMPEG IS INSTALLED
        ---------------------------------------------------------
        Renders an animation of the motion of the particles in 2d.
        :param axis1: first axis to plot.
        :param axis2: second axis to plot.
        :param save: If true, save the animation as an MP4 file.
        :return: animation object. Main purpose is in case the user is using this package in a jupyter notebook.
        """

        # Initialise figure, particle data frames and points to plot
        dfs = {}
        points = []
        fig, ax = plt.subplots(figsize=(8, 6))

        # Put data frames of each particle into the dfs dictionary
        for i in range(int(np.max(self.__database["n"])) + 1):
            p_df = self.return_particle_data(name = i)
            p_df.reset_index(inplace = True)
            dfs[i] = p_df
            #As we want to plot only one dot the arrays in the ax.plot() below must be empty.
            point, = ax.plot([], [], 'o', label = f"n = {i}") #ax.plot returns a tuple of a Line2D object and nothing.
            points.append(point)

        # Set limits on x and y axis to be 0.5 box size to emulate box and better illustrate rebounds
        ax.set_xlim(-self.__box_size/2.0, self.__box_size/2.0)
        ax.set_ylim(-self.__box_size/2.0, self.__box_size/2.0)

        # Set label of axis to the axis passed in
        ax.set_xlabel(axis1)
        ax.set_ylabel(axis2)

        def update_plot(frame):
            """
            Plots specific point onto the figure. Resets when called with different frame.
            Also updates time in animation
            :param frame: integer representing frame of animation from being called by FuncAnimation
            """
            time = dfs[0]["t"][frame]  # Get time from the first dataframe
            for i, df in enumerate(dfs.values()):
                x = df[axis1][frame]
                y = df[axis2][frame]
                points[i].set_data(x, y)
            ax.set_xlabel(f"x\ntime = {time}")
            return points
        
        #Make the animation.
        ani = animation.FuncAnimation(fig, update_plot, frames=len(dfs[0]["t"]), interval=1, blit=True)
        
        #Save the animation in animations folder as mp4 file.
        if save == True:
            file_name = input("Name your file: ")
            ani.save(f'animations/{file_name}_2d.mp4', writer='ffmpeg', fps=25, dpi=100)
        
        #Show animation
        plt.show()

        return ani

    def make_animation_3d(self, save: bool = True) -> animation:
        """
        WARNING: PLEASE MAKE SURE FFMPEG IS INSTALLED
        ---------------------------------------------------------
        Renders an animation of the motion of the particles in 3d.
        :param save: If true, save the animation as an MP4 file.
        :return: animation object. Main purpose is in case the user is using this package in a jupyter notebook.
        """
        # Initialise figure, particle data frames and points to plot
        dfs = {}
        points = []
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Put data frames of each particle into the dfs dictionary
        for i in range(int(np.max(self.__database["n"])) + 1):
            p_df = self.return_particle_data(name = i)
            p_df.reset_index(inplace = True)
            dfs[i] = p_df
            point, = ax.plot([], [], [], 'o')
            points.append(point)
            
        # Set limits on x y z axis to be 0.5 box size to emulate box and better illustrate rebounds
        ax.set_xlim(-self.__box_size/2.0, self.__box_size/2.0)
        ax.set_ylim(-self.__box_size/2.0, self.__box_size/2.0)
        ax.set_zlim(-self.__box_size/2.0, self.__box_size/2.0)
        
        # Set label of axis to the axis passed in
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        def update_plot(frame):
            """
            Plots specific point onto the figure. Resets when called with different frame.
            Also updates time in animation
            :param frame: integer representing frame of animation from being called by FuncAnimation
            """
            time = dfs[0]["t"][frame]  # Get time from the first dataframe
            for i, df in enumerate(dfs.values()):
                points[i].set_data(df["x"][frame], df["y"][frame])
                points[i].set_3d_properties(df["z"][frame]) #has to be used for z axis specifically
            ax.set_xlabel(f"x\ntime = {time}")
            return points
        
        #Make the animation.
        ani = animation.FuncAnimation(fig, update_plot, frames=len(dfs[0]["t"]), interval=1, blit=True)

        #Save the animation in animations folder as mp4 file.
        if save == True:
            file_name = input("Name your file: ")
            ani.save(f'animations/{file_name}_3d.mp4', writer='ffmpeg', fps=25)
        
        #Show animation
        plt.show()

        return ani
    
class SubData(Particles):
    """
    This class holds a database positions, velocities, accelerations, kinetic and potential energies of particles..
    """
    def __init__(self, database_name:pd.DataFrame) -> None:
        """
        Creates a private attribute of the database imported from a file.
        :param database_name (string) : The name the database file in the filesystem.
        :param database_name (pd.DataFrame) : The DataFrame so methods of this class can be used.
        Attributes
        ----------
        __database : Pandas Data Frame
        __parameters : Dictionary of system parameters
        __N, __dt, __steps, __box_size, __duration : Attributes storing each respective system parameter. Prevents running extract_parameters() & casts parameters to correct datatype.
        """
        self.__database = database_name
        self.__steps = len(database_name)
        
    def get_total_kinetic_per_step(self) -> pd.DataFrame:
        """
        Returns pandas DataFrame of the total kinetic energy of the system during each timestep.
        :return: pandas DataFrame of time and total kinetic energy
        """
        if "vx" not in list(self.__database.columns) or "vy" not in list(self.__database.columns) or "vz" not in list(self.__database.columns):
            raise ValueError("Database must have velocity columns for X, Y, Z Components called \"vx\", \"vy\", \"vz\"")
        else:
            Particles.get_total_kinetic_per_step(self)
    
    def get_total_potential_per_step(self) -> pd.DataFrame:
        """
        Returns pandas DataFrame of the total potential energy of the system during each timestep.
        :return: pandas DataFrame of time and total potential energy
        """
        if "PE" not in list(self.__database.columns):
            raise ValueError("Database must have potential energy column \"PE\"")
        else:
            Particles.get_total_potential_per_step(self)

    def plot_relationship(self, col_1: str, col_2: str, hue=None) -> None:
        """
        Returns None if neither parameter is a valid column name.
        Outputs either:
        • countplot (quantity of observations in each categorical bin)
        • catplot (Set to plot box plot to show distribution between categorical var and numeric)
        • rel (scatterplot between col_1 and col_2)
        :param col_1: column 1
        :param col_2: column 2
        :param hue: Grouping variable that will produce elements with different colors. Ideally use a categorical variable.
        :return: None
        """
        if col_1 not in list(self.__database.columns) or col_2 not in list(self.__database.columns):
            print("Enter a valid column name in the database")
            return

        '''
        filter_nan is necessary. 
        The check for categorical variables below tries to convert values in a column to a
        numeric format. If any value cant be converted to numeric, it is NaN. A boolean mask is applied and if entire
        column is False, the column is numeric. However if column has a NaN value before this is even done, the column
        is assumed to be categorical which is bad. 
        '''
        filter_nan = self.__database.dropna(subset=[col_1, col_2])
        if pd.to_numeric(filter_nan[col_1], errors="coerce").notna().all() == False and pd.to_numeric(filter_nan[col_2],
                                                                                                    errors="coerce").notna().all() == False:
            print("Comparing 2 categorical variables")
            sns.countplot(data=self.__database, x=col_1, hue=col_2)
        elif pd.to_numeric(filter_nan[col_1], errors="coerce").notna().all() == False or pd.to_numeric(
                filter_nan[col_2], errors="coerce").notna().all() == False:
            sns.catplot(data=self.__database, x=col_1, y=col_2, kind="box", aspect=1.5, hue=hue)
        else:
            fig, ax = plt.subplots(figsize = (10,6))
            self.__database.plot(ax = ax, x=col_1, y=col_2)
            ax.set_title(f"\"{col_1}\" x \"{col_2}\"", fontsize=16)

