import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy import stats

st.set_page_config(
     page_title='Dashboard',
     layout="wide",
     initial_sidebar_state="expanded",
     page_icon="agilogo(1).jpeg"
)

#set sidebar
st.sidebar.image("agilogo(1).jpeg", use_column_width=True)
st.sidebar.title('Agricultural Institute Dashboard')
st.sidebar.divider()

st.markdown("# Grades Evaluation")
st.markdown("### From seeing current grade distributions to obtaining a list on students that need academic help, this dashboard will help you do just that!")


#data upload
uploaded_file = st.file_uploader("Upload grade dataset here (csv file)")
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, encoding='utf-8', thousands=None)
        # Remove formatting from numerical columns
        for col in data.select_dtypes(include='number'):
            data[col] = data[col].astype(str).str.replace(',', '') #makes it so it doesnt add commas to the data
        st.write(data)
    except Exception as e:
        st.error("Error loading data")
st.write("See if the data loaded in is correct!")

data['Unit Taken'] = pd.to_numeric(data['Unit Taken'])

unique_ids_count = len(data['ID'].unique()) #counts the number of students in the dataset
st.write("There are ", unique_ids_count, " students in the dataset")
st.sidebar.metric(label="Number of Students", value=unique_ids_count) #adds the metrics to the sidebar
st.sidebar.divider() #adds a line underneath it







#variable additions
data1 = (data.groupby('ID').apply(lambda x: x.assign(totalUnits=x['Unit Taken'].sum(), coursesTaken=x['Term'].count())).reset_index(drop=True))

# Move and rename columns
data2 = data1.copy()
data2 = (data2.assign(classes=lambda x: x['Subject'] + x['Catalog']).rename(columns={'Unit Taken': 'units'}))

# Map grades
data2['Grade2'] = data2['Grade'].map({'A+': 'A', 'A': 'A', 'A-': 'A', 
                                           'B+': 'B', 'B': 'B', 'B-': 'B',
                                           'C+': 'C', 'C': 'C', 'C-': 'C',
                                           'D+': 'D', 'D': 'D', 'D-': 'D',
                                           'F': 'F'}).fillna('IN')








#create tables to show frequencies per factor
table1 = pd.crosstab(data2['classes'], data2['EarnCredit'])
table2 = pd.crosstab(data2['classes'], data2['Grade'])
table3 = pd.crosstab(data2['classes'], data2['Grade2'])

# Convert tables to DataFrames
table1_df = pd.DataFrame(table1.stack()).reset_index()
table1_df.columns = ['Class', 'EarnCredit', 'Frequency']
table2_df = pd.DataFrame(table2.stack()).reset_index()
table2_df.columns = ['Class', 'Grade', 'Frequency']
table3_df = pd.DataFrame(table3.stack()).reset_index()
table3_df.columns = ['Class', 'Grade', 'Frequency']




st.markdown("# Student Grades")
st.write("If the charts don't show up, press refresh!")



st.markdown("### Histogram of Grades by Earned Credit")
# Set up the FacetGrid
g = sns.FacetGrid(table1_df, col='Class', col_wrap=3, height=3, aspect=1.25)
# Plotting on each facet
g.map(sns.barplot, x='EarnCredit', y='Frequency', data=table1_df.reset_index(), hue='EarnCredit',palette='Reds', errorbar = None)
# Accessing each subplot and annotating bars
for ax in g.axes.flat:
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 2),
                    textcoords='offset points')
# Set title, x-label, and y-label for each facet
g.set_titles('Histogram of Earn Credit by Class - {col_name}', fontsize=24)
g.set_axis_labels('Earn Credit', 'Frequency')
# Add legend
g.add_legend()
# Get the figure and axes objects
fig = plt.gcf()
# Display the plot in Streamlit
st.pyplot(fig)






st.markdown("### Histogram of Grades by Class")
# Set up the FacetGrid
g = sns.FacetGrid(table3_df, col='Class', col_wrap=3, height=3, aspect=1.25)
# Plotting on each facet
g.map(sns.barplot, x='Grade', y='Frequency', data=table3_df.reset_index(), hue='Grade',palette='Reds', errorbar=None)
# Accessing each subplot and annotating bars
for ax in g.axes.flat:
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 2),
                    textcoords='offset points')
# Set title, x-label, and y-label for each facet
g.set_titles('Histogram of Earn Credit by Class - {col_name}', fontsize=24)
g.set_axis_labels('Grade', 'Frequency')
# Add legend
g.add_legend()
# Get the figure and axes objects
fig = plt.gcf()
# Display the plot in Streamlit
st.pyplot(fig)





st.markdown("### Histogram of Grades by Class -- In-depth")
# Set up the FacetGrid
g = sns.FacetGrid(table2_df, col='Class', col_wrap=3, height=3, aspect=1.25)
# Plotting on each facet
g.map(sns.barplot, x='Grade', y='Frequency', data=table2_df.reset_index(), hue='Grade',palette='Reds', errorbar = None)
# Accessing each subplot and annotating bars
for ax in g.axes.flat:
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 2),
                    textcoords='offset points')
# Set title, x-label, and y-label for each facet
g.set_titles('Histogram of Earn Credit by Class - {col_name}', fontsize=24)
g.set_axis_labels('Grade', 'Frequency')
# Add legend
g.add_legend()
# Get the figure and axes objects
fig = plt.gcf()
# Display the plot in Streamlit
st.pyplot(fig)







#create new variables for clustering

# Define a function to calculate grade points
def calculate_grade_point(grade):
    if grade == "A+":
        return 13
    elif grade == "A":
        return 12
    elif grade == "A-":
        return 11
    elif grade == "B+":
        return 10
    elif grade == "B":
        return 9
    elif grade == "B-":
        return 8
    elif grade == "C+":
        return 7
    elif grade == "C":
        return 6
    elif grade == "C-":
        return 5
    elif grade == "D+":
        return 4
    elif grade == "D":
        return 3
    elif grade == "D-":
        return 2
    elif grade == "F":
        return 1
    else:
        return 0
# Apply the function to create a new column 'gradePoint'
data3 = data2.copy()  # Make a copy to avoid modifying the original DataFrame
data3['gradePoint'] = data3['Grade'].apply(calculate_grade_point)
# Group by 'ID' and calculate average and sum of grade points
data3['sumGradePoint'] = data3.groupby('ID')['gradePoint'].transform('sum')
data3['avgGradePoint'] = data3.groupby('ID')['gradePoint'].transform('mean')
data3['avgGradePoint'] = data3['avgGradePoint']/3
# Ungroup the DataFrame
data3 = data3.reset_index(drop=True)




st.markdown("# At Risk Students")
# Create histogram plot
fig, ax = plt.subplots(figsize=(10, 3))  # Set figure size here
sns.histplot(data3["avgGradePoint"], ax=ax, color = "red")
ax.set_title('Histogram of Average Grade Point')
ax.set_xlabel('Average Grade Point')
ax.set_ylabel('Frequency')
# Display the plot in Streamlit
st.pyplot(fig)




st.write("The plot above lets us know the distribution of students 'GPA's'")
st.write("In this case, the student 'GPA's' is a  calculated number based on the number of classes a student has taken and their grades in those classes")
st.write("Down below is a clustering analysis that tells us where students lie per academic status -- Great Standing, Good Standing, and At Risk ")




#cluster 2
#  Prepare data for clustering (reshaping into a 2D array)
X = data3['avgGradePoint'].values.reshape(-1, 1)
# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
# Create the scatter plot
fig, ax = plt.subplots(figsize=(8, 3))  # Set figure size here
scatter = ax.scatter(data3.index, data3['avgGradePoint'], c=kmeans.labels_, cmap='Dark2')
# Plot centroids
centroids = kmeans.cluster_centers_
for centroid in centroids:
    ax.plot([0, len(data3)], [centroid, centroid], c='red', linestyle='--', linewidth=2)
# Set plot title and labels
ax.set_title('Clustering of Average Grade Point')
ax.set_xlabel('Student ID')
ax.set_ylabel('Average Grade Point')
# Add legend
legend = ax.legend(*scatter.legend_elements(), title='Cluster', loc='upper right')
ax.add_artist(legend)
# Display the plot in Streamlit
st.pyplot(fig)




for i, center in enumerate(kmeans.cluster_centers_):
    rounded_center = round(center[0], 2)
    st.write(f"##### Cluster {i+1} = Average grade point of {rounded_center}")
cluster_centers = kmeans.cluster_centers_
cluster_centers = np.array(cluster_centers)
# Calculate the minimum cluster center
min_cluster_center = round(cluster_centers.min(),2)
st.write("##### Students of concern have an average grade point of around ",min_cluster_center)




st.markdown("## Labeling students of concern" )
data4 = data3.copy()
data4['cluster'] = kmeans.labels_ #create new variable
data4['cluster'] = data4['cluster'].replace({2: 'Great Standing', 0: 'Good Standing', 1: 'At Risk'}) #relabel numeric with string
data4 = data4[["ID","Name","cluster","totalUnits","coursesTaken","gradePoint","sumGradePoint","avgGradePoint"]] #subset columns
#print a blurb telling how many students are at risk
clusterData = data4[data4["cluster"] == "At Risk"] 
unique_ids_count = len(clusterData["ID"].unique())
st.write("There are ", unique_ids_count, " students that have academic warning")
#assigns risk percentile
percentile = stats.percentileofscore(data4['avgGradePoint'],data4['avgGradePoint'])
data4['riskPercentile'] = np.round(100-percentile,0)
data4 = data4.groupby('ID').first().reset_index()

#sidebar add in
st.sidebar.metric(label="Students of Concern", value=unique_ids_count)
st.sidebar.divider()


#sidebar add-in -- pull list of people with really low percentiles
data5 = data4[data4['riskPercentile']>=95]
data5 = data5.sort_values(by='riskPercentile', ascending=False)
data5 = data5[["Name","riskPercentile"]]
# Display top 10 high-risk students in the sidebar
st.sidebar.title("High Risk Students:")
st.sidebar.write(data5)






#got function below from here: 
#https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")
    if not modify:
        return df
    df = df.copy()
    modification_container = st.container()
    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
    return df
#Write dataframe
st.write("Use this table to filter the 3 different academic clusters and print out a csv file with students name and their info")
st.write("_Instructions: Click 'Add filters' → filter the dataframe on 'cluster' → pull up the cluster you want to see_")
data4 = filter_dataframe(data4)
st.dataframe(data4)


#download csv
@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')
csv_filter = convert_df(data4)
st.download_button(
   "Press to Download filtered data from above",
   csv_filter,
   "file.csv",
   "text/csv",
   key='download-csv'
)



#pull grades from classes of filtered students

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add student names")
    if not modify:
        return df
    df = df.copy()
    modification_container = st.container()
    with modification_container:
        to_filter_columns = st.multiselect("List Students names", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            else:
                user_text_input = right.text_input(
                    f"Names for {column} (comma-separated)",
                )
                if user_text_input:
                    names = [name.strip() for name in user_text_input.split(',')]
                    df = df[df[column].isin(names)]
    return df

data3 = data3[["ID","Name","classes","Descr","Term","Session","Section","Class Nbr"]]
st.markdown("### Use this table to see which classes students have fared in")
st.write("_Instructions: click 'add student names' → click 'Name' from the drop down → type in students name (Full Name as shown in dataset)_")
data3 = filter_dataframe(data3)
st.dataframe(data3)


#data download for filtered students
@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')
csv_filter = convert_df(data3)
st.download_button(
   "Press to Download students data from above",
   csv_filter,
   "file.csv",
   "text/csv",
   key='download-csv-student'
)
