import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PlotBasic:
    def plot_scatter(self, df, y_data, y_feat, x_data, x_feat, x_label, y_label):
        y_col = f"{y_data}_{y_feat}" if y_feat else y_data
        x_col = f"{x_data}_{x_feat}" if x_feat else x_data
        data = df[[x_col, y_col, 'country']].dropna()
        y = data[y_col]
        X = data[x_col]
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        print(results.summary())
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data, x=x_col, y=y_col, hue = 'country')
        sns.lineplot(x=data[x_col], y=results.fittedvalues, color='red', label=f'Regression Line (R-squared: {results.rsquared:.2f})')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'{y_label} vs {x_label}')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_histogram(self, df, data_col, feature=None, label='Data', title='Histogram'):
        col_name = f"{data_col}_{feature}" if feature else data_col
        bins, color, edgecolor =  10, 'skyblue', 'black'
        plt.figure(figsize=(8, 6))
        sns.histplot(df, x = col_name, bins=bins, color=color, hue = 'country', edgecolor=edgecolor)
        plt.xlabel(label)
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(axis='y', alpha=0.75)
        plt.legend()
        plt.show()

    def plot_timeseries(self, df, y_data, y_feat, x_data, x_feat, x_label, y_label):
        y_col = f"{y_data}_{y_feat}" if y_feat else y_data
        x_col = f"{x_data}_{x_feat}" if x_feat else x_data
        
        fig, ax1 = plt.subplots()

        df[x_col].plot(ax=ax1, color='blue')
        ax1.set_xlabel('Year')
        ax1.set_ylabel(x_col, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        df[y_col].plot(ax=ax2, color='red')
        ax2.set_ylabel(y_col, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        plt.title(f'{ y_col} vs {x_col}')
        plt.show()