import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('C:/Users/THILAK.S/Downloads/Data P3 MeriSKILL/HR-Employee-Attrition.csv')


df.head





df.tail(5)



df.shape


df.columns


#Data reading, checking dimensions and information of the data
print(df)

print('dimensions:')
print(df.shape)

print('Information:')
df.info()


print(df.apply(lambda col: col.unique())) 


df.nunique()


df.corr()


#Cheking for duplicates 
value=len(df[df.duplicated()])
print(value) 


#Cheking for missing
df.isnull().sum()


#Statistical summary
df.describe().T


df.drop(['EmployeeCount'],axis=1,inplace=True)
df.drop(['Education'],axis=1,inplace=True)
df.drop(['RelationshipSatisfaction'],axis=1,inplace=True)
df.drop(['StockOptionLevel'],axis=1,inplace=True)
df.drop(['TrainingTimesLastYear'],axis=1,inplace=True)
df.drop(['WorkLifeBalance'],axis=1,inplace=True)
df.drop(['StandardHours'],axis=1,inplace=True)
df.drop(['YearsWithCurrManager'],axis=1,inplace=True)



df.head(5)



def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="upper right"):
    """
    Plot stacked bars with annotations
    """
    ax = dataframe.plot(kind="bar", stacked=True, figsize=size_, rot=rot_, title=title_)

    # Annotate bars
    annotate_stacked_bars(ax, textsize=14)
    # Rename legend
    plt.legend(["Retention", "Attrition"], loc=legend_)
    # Labels
    plt.ylabel("Company base (%)")
    plt.show()

def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=13):
    """
    Add value annotations to the bars
    """
    # Iterate over the plotted rectangles/bars
    for p in ax.patches:
        # Calculate annotation
        value = str(round(p.get_height(), 1))
        # If value is 0 do not annotate
        if value == '0.0':
            continue
        ax.annotate(
            value,
            ((p.get_x() + p.get_width() / 2) * pad - 0.05, (p.get_y() + p.get_height() / 2) * pad),
            color=colour,
            size=textsize
        )


Attrition  = df [['Age', 'Attrition']]
Attrition.columns = ['TotalWorkingYear', 'Attrition']
attrition_total = Attrition.groupby(Attrition['Attrition']).count()
attrition_percentage = attrition_total / attrition_total.sum() * 100


plot_stacked_bars(attrition_percentage.transpose(), "Attrition status", (5, 5),legend_="lower right")




import plotly.graph_objects as go
gender_counts = df['Gender'].value_counts()

colors = ['SkyBlue', 'Lightgreen']  # Specify the colors you want to use

fig = go.Figure(data=[go.Pie(labels=gender_counts.index, values=gender_counts, hole=0.4, marker=dict(colors=colors))])

fig.update_layout(
    title='Gender Distribution'
)

fig.show()



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))  

sns.lineplot(data=df, x='YearsInCurrentRole', y='MonthlyIncome')
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))  # Adjust the figsize according to your desired size

sns.histplot(data=df, x='Age', hue='Attrition', kde=True, palette={'Yes': 'red', 'No': 'blue'})

plt.title('Distribution of Age by Attrition')
plt.xlabel('Age')
plt.ylabel('Density')

plt.show()



fig = px.pie(df, names='MaritalStatus', title='Marital Status', color_discrete_sequence=['#48795E', '#003566', '#707BAD'])
fig.show()


fig = px.histogram(df, x='MaritalStatus', color='Attrition', title='Number of attritional employees or not by Marital Status')
fig.update_layout(
    xaxis_title='Marital Status',
    yaxis_title='Number of employees',
    yaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=100
    )
)
fig.update_layout(template="plotly_dark" )
fig.show()



import plotly.graph_objects as go

counts = df.groupby(['JobSatisfaction', 'Attrition']).size().unstack().reset_index()

fig = go.Figure()
fig.add_trace(go.Bar(y=counts['JobSatisfaction'], x=counts['Yes'], name='Attrition: Yes', orientation='h', marker_color='Yellow'))
fig.add_trace(go.Bar(y=counts['JobSatisfaction'], x=counts['No'], name='Attrition: No', orientation='h', marker_color='pink'))

fig.update_layout(
    title='Count of Employees by Job Satisfaction and Attrition',
    yaxis_title='Job Satisfaction',
    xaxis_title='Count',
    barmode='group'
)
fig.update_layout(template="plotly_dark" )

fig.show()



def barplot(column, horizontal):
    plt.figure(figsize=(10, 7))
    sns.countplot(x=column, data=df, palette='viridis')
    plt.xlabel(column)
    plt.ylabel("Attrition")
    plt.title(f"Count of {column}", fontweight='bold')
    plt.xticks(rotation=45)
    sns.despine()
    plt.tight_layout()
    plt.show()

barplot('Department', True)

