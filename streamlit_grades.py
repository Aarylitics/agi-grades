import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(
     page_title='Dashboard',
     layout="wide",
     initial_sidebar_state="expanded",
     page_icon="agilogo.jpeg"
)

#set sidebar
st.sidebar.image("agilogo.jpeg", use_column_width=True)
st.sidebar.title('Agriculture Institute Dashboard')
st.sidebar.divider()

st.markdown("# Grades Evaluation")
st.write("Interested in what the current grades look like? Or how about who's still ")

uploaded_file = st.file_uploader("Upload grade dataset here (csv file)")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, encoding='utf-8')
        st.write(data)
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.write("See if the data loaded in is correct!")

unique_ids_count = len(data['ID'].unique())
st.write("There are ", unique_ids_count, " students in the dataset")

st.sidebar.metric(label="Number of Students", value=unique_ids_count)
st.sidebar.divider()

#print a table that displays names and id's?

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

table1 = pd.crosstab(data2['classes'], data2['EarnCredit'])

# Selecting classes and Grade
table2 = pd.crosstab(data2['classes'], data2['Grade'])

# Selecting classes and Grade2
table3 = pd.crosstab(data2['classes'], data2['Grade2'])

# Assuming table1, table2, and table3 are the DataFrames obtained from the crosstabs

# Convert tables to pandas DataFrames
table1_df = pd.DataFrame(table1.stack()).reset_index()
table1_df.columns = ['Class', 'EarnCredit', 'Frequency']

table2_df = pd.DataFrame(table2.stack()).reset_index()
table2_df.columns = ['Class', 'Grade', 'Frequency']

table3_df = pd.DataFrame(table3.stack()).reset_index()
table3_df.columns = ['Class', 'Grade', 'Frequency']

st.markdown("# Graphs")
st.write("Each graph will have a corresponding table underneath it. I recommend you look at them to get a more accurate number!")

st.markdown("### Histogram by Credit Earned")
# Set up the FacetGrid
g = sns.FacetGrid(table1_df, col='Class', col_wrap=3,height = 3, aspect = 1.25)

# Plotting on each facet
g.map_dataframe(sns.barplot, x='EarnCredit', y='Frequency', hue ='EarnCredit')

# Set title, x-label, and y-label for each facet
g.set_titles('Histogram of Earn Credit by Class - {col_name}',fontsize=24)  # Include {col_name} to display the factor being graphed
g.set_xlabels('Earn Credit', fontsize=12)
g.set_ylabels('Frequency', fontsize=12)

# Display the plot in Streamlit
st.pyplot(plt.gcf())

st.write("History of Earn Credit by Class:")
st.write(table1)

st.markdown("### Histogram of Grades by Class")
# Set up the FacetGrid
g = sns.FacetGrid(table3_df, col='Class', col_wrap=3, height = 3, aspect = 1.25)

# Plotting on each facet
g.map_dataframe(sns.barplot, x='Grade', y='Frequency', hue='Grade')

# Set title, x-label, and y-label for each facet
g.set_titles('Histogram of Grades by Class - {col_name}')
g.set_xlabels('Grade')
g.set_ylabels('Frequency')

# Display the plot in Streamlit
st.pyplot(plt.gcf())

st.write("\nHistogram of Grades by Class:")
st.write(table3)


st.markdown("### Histogram of Grades by Class -- In-depth")
# Set up the FacetGrid
g = sns.FacetGrid(table2_df, col='Class', col_wrap=3,height = 3, aspect = 1.25)

# Plotting on each facet
g.map_dataframe(sns.barplot, x='Grade', y='Frequency', hue='Grade')

# Set title, x-label, and y-label for each facet
g.set_titles('Histogram of Grades by Class - {col_name}')
g.set_xlabels('Grade')
g.set_ylabels('Frequency')

# Display the plot in Streamlit
st.pyplot(plt.gcf())

st.write("\nHistorgram of Grades by Class:")
st.write(table2)

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

# Ungroup the DataFrame
data3 = data3.reset_index(drop=True)

st.markdown("# Students of Concern")

# Create histogram plot
fig, ax = plt.subplots(figsize=(10, 3))  # Set figure size here
sns.histplot(data3["avgGradePoint"], ax=ax)
ax.set_title('Histogram of Average Grade Point')
ax.set_xlabel('Average Grade Point')
ax.set_ylabel('Frequency')

# Display the plot in Streamlit
st.pyplot(fig)

st.write("The plot above lets us know the distribution of students 'GPA's'")
st.write("In this case, the student 'GPA's' is a  calculated number based on the number of classes a student has taken and their grades in those classes")

st.write("Down below is a clustering analysis that tells us where students lie per academic status -- Great Standing, Good Standing, Decent Standing, and Academic Warning ")

#cluster1
#  Prepare data for clustering (reshaping into a 2D array)
X = data3['avgGradePoint'].values.reshape(-1, 1)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

# Plot the data points
fig, ax = plt.subplots(figsize=(10, 3))
scatter = ax.scatter(data3['sumGradePoint'],data3['coursesTaken'],c=kmeans.labels_)
ax.set_title('Identifying Students of Concern')
ax.set_ylabel('Number of Courses Taken')
ax.set_xlabel('Sum Grade Point')

# Add legend
legend = ax.legend(*scatter.legend_elements(), title='Cluster', loc='upper right')
ax.add_artist(legend)

# Display the plot in Streamlit
st.pyplot(fig)

#  Prepare data for clustering (reshaping into a 2D array)
X = data3['avgGradePoint'].values.reshape(-1, 1)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

# Create the scatter plot
fig, ax = plt.subplots(figsize=(8, 3))  # Set figure size here
ax.scatter(data3.index, data3['avgGradePoint'], c=kmeans.labels_, cmap='viridis')

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
    st.write(f"Cluster {i+1} = Average grade point of {rounded_center}")

cluster_centers = kmeans.cluster_centers_
cluster_centers = np.array(cluster_centers)

# Calculate the minimum cluster center
min_cluster_center = round(cluster_centers.min(),2)
st.write("Students of concern have an average grade point of around ",min_cluster_center)

st.markdown("## Labeling students of concern" )
data4 = data3.copy()
data4['cluster'] = kmeans.labels_

data4['cluster'] = data4['cluster'].replace({1: 'Great Standing', 0: 'Good Standing', 3: 'Decent Standing', 2: 'Academic Warning'})

data4 = data4[["ID","Name","cluster","totalUnits","coursesTaken","gradePoint","sumGradePoint","avgGradePoint"]]

clusterData = data4[data4["cluster"] == "Academic Warning"] 
unique_ids_count = len(clusterData["ID"].unique())
st.write("There are ", unique_ids_count, " students that have academic warning")

data4 = data4.groupby('ID').first().reset_index()

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
st.write("Use this table to filter the 4 different academic clusters")
st.write("Instructions: Click 'Add filters' --> filter the dataframe on 'cluster' --> pull up the cluster you want to see")
data5 = filter_dataframe(data4)


st.dataframe(data5)

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

csv_filter = convert_df(data5)

st.download_button(
   "Press to Download filtered data from above",
   csv_filter,
   "file.csv",
   "text/csv",
   key='download-csv'
)

#potentially pull grades from classes of filtered students?

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
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

data6 = data3.copy()
data6['cluster'] = kmeans.labels_
data6['cluster'] = data6['cluster'].replace({1: 'Great Standing', 0: 'Good Standing', 3: 'Decent Standing', 2: 'Academic Warning'})
data6 = data6[["ID","Name","cluster","classes","Descr","Term","Session","Section","Class Nbr"]]

st.markdown("### Use this table to see which classes students have fared in")
st.write("Instructions: click 'add student names' --> click 'Name' from the drop down --> type in students name")
data6 = filter_dataframe(data6)


st.dataframe(data6)
st.sidebar.metric(label="Students of Concern", value=unique_ids_count)
st.sidebar.divider()
#want to add color to sidebar

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

csv_filter = convert_df(data6)

st.download_button(
   "Press to Download students data from above",
   csv_filter,
   "file.csv",
   "text/csv",
   key='download-csv-student'
)

