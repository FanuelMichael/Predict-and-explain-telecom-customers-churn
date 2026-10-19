import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ------------------------------- Exploratory Data Analysis (EDA) ------------------------------
eda_plot_df,num_cols,cat_cols= None,None,None
def data_analysis(file_path):
    # 1. Load the dataset
    df = pd.read_csv(file_path)

    print("Number of customers, Number of columns/features: " + str(df.shape))
    print("\nDataset features data types:")
    print(df.info())

    print("\nDataset first 5 rows:\n" + str(df.head()))
    print("\nDataset bottom 5 rows:\n" + str(df.tail()))

    df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # --------------numerical and categorical columns----------------------
    global num_cols,cat_cols
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print("\nNumerical columns:\n" + str(num_cols))
    print("\nCategorical columns:\n" + str(cat_cols))
    print("\nDataset features unique values:\n" + str(df.nunique()))
    print("\nDataset summary statistics of numerical columns\n" + str(df.describe()))
    global eda_plot_df
    eda_plot_df = df.copy()

    # ------------------------Method calls for designing Plots-------------------------------
    plot_plots_recursive(step=1, df=eda_plot_df, num_cols=num_cols)
    plot_categorical_plots(cat_cols, eda_plot_df, 0, 3, 7)
    print("\nFor the graphical exploratory dataset analysis read inside a file called data/data_analysis.\n")

def plot_plots_recursive(step, df, num_cols, max_steps=2):
    # Base case: stop after Plot 2
    if step > max_steps:
        return
    # ---------------------- Plot 1 ----------------------
    if step == 1:
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))
        plt.suptitle('Teleco Customer Churn Analysis Plot',fontsize=12, fontweight='bold', y=0.98)
        # Plot 1
        sns.countplot(ax=axes[0], x='Churn', data=df, palette='Set2')
        axes[0].set_title('1. Overall churn distribution', fontsize=10, pad=5)
        axes[0].set_xlabel('Churn', fontsize=10)

        # Plot 2
        sns.countplot(ax=axes[1], x='SeniorCitizen', data=df, palette='Set2')
        axes[1].set_title('2. Senior Citizen distribution by churn', fontsize=10, pad=5)
        axes[1].set_xlabel('Senior Citizen', fontsize=10)

        # Plot 3
        sns.heatmap(ax=axes[2],data=df[num_cols].corr(),annot=True,cmap='coolwarm',center=0,fmt='.2f',linewidths=0.5)
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
        axes[2].set_title('3. Correlation Heatmap of Numerical Features',fontsize=10, pad=5)

        plt.tight_layout(rect=(0, 0.03, 1, 0.94))
        plt.show()
    # ---------------------- Plot 2 ----------------------
    if step == 2:
        fig, axes1 = plt.subplots(1, 3, figsize=(14, 6))
        plt.suptitle('Teleco Customer Churn Analysis Plot',fontsize=14, fontweight='bold', y=0.98)

        # Plot 4
        sns.kdeplot(ax=axes1[0],data=df,x='MonthlyCharges',hue='Churn',fill=True,common_norm=False,palette='coolwarm',alpha=0.5)
        axes1[0].set_title('4: Monthly Charges Density by Churn',fontsize=10, pad=5)
        axes1[0].set_xlabel('Monthly Charges', fontsize=10)

        # Plot 5
        sns.kdeplot(ax=axes1[1],data=df,x='TotalCharges',hue='Churn',fill=True,common_norm=False,palette='coolwarm',alpha=0.5)
        axes1[1].set_title('5: Total Charges Density by Churn',fontsize=10, pad=5)
        axes1[1].set_xlabel('Total Charges', fontsize=10)

        # Plot 6
        sns.kdeplot(ax=axes1[2],data=df,x='tenure',hue='Churn',fill=True,common_norm=False,palette='crest',alpha=0.5)
        axes1[2].set_title('6: Tenure Density by Churn',fontsize=10, pad=5)
        axes1[2].set_xlabel('Tenure (months)', fontsize=10)

        plt.tight_layout(rect=(0, 0.03, 1, 0.94))
        plt.show()
    # Recursive call for the next Plot
    plot_plots_recursive(step + 1, df, num_cols, max_steps)

def plot_categorical_plots(cat_cols, df, start_index=0, plot_num=3, max_Plots=7):
    # Base case: stop when Plot_num exceeds max_Plots
    if plot_num > max_Plots:
        return

    # Select the next 3 categorical columns
    cols_slice = cat_cols[start_index:start_index + 3]
    # If fewer than 3 columns remain, stop recursion
    if len(cols_slice) < 3:
        return

    # Create Plot (1 × 3 grid)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
    plt.suptitle(f'Teleco Customer Churn Analysis Plot',fontsize=14, fontweight='bold', y=0.98)

    # Plot each of the 3 categorical columns
    for i, col in enumerate(cols_slice):
        sns.countplot(ax=axes[i], data=df, x=col, hue="Churn", palette='Set2')
        axes[i].set_title(f'{col} Distribution by Churn', fontsize=10, pad=5)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].tick_params(axis='x', rotation=15)

    plt.tight_layout(rect=(0, 0.03, 1, 0.94))
    plt.show()
    # Recursive call for the next Plot
    plot_categorical_plots(cat_cols,df,start_index=start_index + 3,plot_num=plot_num + 1,max_Plots=max_Plots)
