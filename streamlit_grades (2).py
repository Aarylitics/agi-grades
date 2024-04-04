import streamlit as st 
import pandas as pd

st.markdown("# Grades Evaluation")

st.write("Interested in what the current grades look like? Or how about who's still ")

uploaded_file = st.file_uploader("Upload grade dataset here")

if uploaded_file is not None:
     data = pd.read_csv(uploaded_file, encoding='utf-8')
     st.write(data)

unique_ids_count = len(data['ID'].unique())
st.write("There are ", unique_ids_count, " students in the dataset")

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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming table1, table2, and table3 are the DataFrames obtained from the crosstabs

# Convert tables to pandas DataFrames
table1_df = pd.DataFrame(table1.stack()).reset_index()
table1_df.columns = ['Class', 'EarnCredit', 'Frequency']

table2_df = pd.DataFrame(table2.stack()).reset_index()
table2_df.columns = ['Class', 'Grade', 'Frequency']

table3_df = pd.DataFrame(table3.stack()).reset_index()
table3_df.columns = ['Class', 'Grade', 'Frequency']

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(15, 30))

# Plot for table1
sns.barplot(data=table1_df, x='EarnCredit', y='Frequency', hue='EarnCredit', ax=axes[0])
axes[0].set_title('Histogram of Earn Credit by Class',fontsize=24)
axes[0].set_xlabel('Earn Credit',fontsize=24)
axes[0].set_ylabel('Frequency',fontsize=24)

for index, row in table1_df.iterrows():
    axes[0].annotate(f"{row['Frequency']}", (row.name, row['Frequency']), ha='center', va='center_baseline',fontsize=24)

# Plot for table2
sns.barplot(data=table2_df, x='Grade', y='Frequency', hue='Grade', ax=axes[1])
axes[1].set_title('Histogram of Grades by Class',fontsize=24)
axes[1].set_xlabel('Grade',fontsize=24)
axes[1].set_ylabel('Frequency',fontsize=24)

for index, row in table2_df.iterrows():
    axes[1].annotate(f"{row['Frequency']}", (row.name, row['Frequency']), ha='center', va='center',fontsize=24)

# Plot for table3
sns.barplot(data=table3_df, x='Grade', y='Frequency', hue='Grade', ax=axes[2])
axes[2].set_title('Histogram of Grades by Class',fontsize=24)
axes[2].set_xlabel('Grade',fontsize=24)
axes[2].set_ylabel('Frequency',fontsize=24)

for index, row in table3_df.iterrows():
    axes[2].annotate(f"{row['Frequency']}", (row.name, row['Frequency']), ha='center', va='baseline',fontsize=24)

# Display the plots
st.pyplot(fig)

st.write("Resort to the tables below to see a clearer number!")

# Displaying the tables
st.write("History of Earn Credit by Class:")
st.write(table1)
st.write("\nHistorgram of Grades by Class:")
st.write(table2)
st.write("\nHistogram of Grades by Class:")
st.write(table3)

# Set up the FacetGrid
g = sns.FacetGrid(table1_df, col='Class', col_wrap=3)

# Plotting on each facet
g.map_dataframe(sns.barplot, x='EarnCredit', y='Frequency', hue ='EarnCredit')

# Set title, x-label, and y-label for each facet
g.set_titles('Histogram of Earn Credit by Class')
g.set_xlabels('Earn Credit')
g.set_ylabels('Frequency')

# Add annotations to each facet
for ax in g.axes.flatten():
    for index, row in table1_df.iterrows():
        if row['Class'] == ax.get_title():
            ax.annotate(f"{row['Frequency']}", (row.name, row['Frequency']), ha='center', va='center_baseline', fontsize=12)

# Display the plot in Streamlit
st.pyplot(plt.gcf())

# Set up the FacetGrid
g = sns.FacetGrid(table2_df, col='Class', col_wrap=3)

# Plotting on each facet
g.map_dataframe(sns.barplot, x='Grade', y='Frequency', hue='Grade')

# Set title, x-label, and y-label for each facet
g.set_titles('Histogram of Grades by Class')
g.set_xlabels('Grade')
g.set_ylabels('Frequency')

# Add annotations to each facet
for ax in g.axes.flatten():
    for index, row in table2_df.iterrows():
        if row['Class'] == ax.get_title():
            ax.annotate(f"{row['Frequency']}", (row.name, row['Frequency']), ha='center', va='center', fontsize=12)

# Display the plot in Streamlit
st.pyplot(plt.gcf())


# Set up the FacetGrid
g = sns.FacetGrid(table3_df, col='Class', col_wrap=3, sharey=False)

# Plotting on each facet
g.map_dataframe(sns.barplot, x='Grade', y='Frequency', hue='Grade')

# Set title, x-label, and y-label for each facet
g.set_titles('Histogram of Grades by Class')
g.set_xlabels('Grade')
g.set_ylabels('Frequency')

# Add annotations to each facet
for ax in g.axes.flatten():
    for index, row in table3_df.iterrows():
        if row['Class'] == ax.get_title():
            ax.annotate(f"{row['Frequency']}", (row.name, row['Frequency']), ha='center', va='baseline', fontsize=12)

# Display the plot in Streamlit
st.pyplot(plt.gcf())

# Assuming data2 is a pandas DataFrame

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

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

#cluster1

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0).fit(data3[['avgGradePoint','coursesTaken']])

# Plot the data points
fig, ax = plt.subplots()
scatter = ax.scatter(data3['avgGradePoint'],data3['coursesTaken'],c=kmeans.labels_)
ax.set_title('Identifying Students of Concern')
ax.set_ylabel('Number of Courses Taken')
ax.set_xlabel('Average Grade Point')

# Add cluster centers to the plot
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='s', s=100, label='Centroids')

# Add legend
legend = ax.legend(*scatter.legend_elements(), title='Cluster', loc='upper right')
ax.add_artist(legend)

# Display the plot in Streamlit
st.pyplot(fig)

st.write("Cluster Centers:", kmeans.cluster_centers_)
