import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import google.generativeai as genai
from io import BytesIO
import asyncio
import io
st.set_page_config(page_title="Data Insights Toolkit", page_icon="📊")

genai.configure(api_key= st.secrets["GOOGLE_API_KEY"])
st.markdown("""
    <style>
        .header_style {
            font-size: 56px;
            color: #1E90FF; /* Change the color as per your preference */
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2); /* Add shadow effect */
        }
        .response_style {
            font-size: 54px;
            color: green; /* Change the color as per your preference */
            text-align: left;
            margin-bottom: 20px;
        }
        .ouput {
            font-size: 29px;
            color: #333333; /* Change the color as per your preference */
            line-height: 1.5;
            padding: 10px;
            border-radius: 5px;
            background-color: #f0f0f0; /* Change the background color as per your preference */
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)
created_style = """
    color: #888888; /* Light gray color */
    font-size: 99px; /* Increased font size */
"""
header_style = """
    text-align: center;
    color: white;
    background-color: #800080;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 30px;
"""
def upload_data():
    """Uploads a CSV file to be used for analysis."""
    uploaded_file = st.file_uploader("Upload data in CSV format", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())
        return data
    else:
        return pd.DataFrame()

def univariate_plot(plot_type, data, column_name):

       
    plt.figure(figsize=(8, 6))
    
    if plot_type == "Histogram":
        sns.histplot(data=data, x=column_name, kde=False)
    elif plot_type == "Kernel Density Estimate (KDE)":
        sns.kdeplot(data=data, x=column_name)
    elif plot_type == "Box Plot":
        sns.boxplot(data=data, x=column_name)
    elif plot_type == "Rug Plot":
        sns.rugplot(data=data, x=column_name)
    elif plot_type == "ECDF Plot":
        sns.ecdfplot(data=data, x=column_name)
    elif plot_type == "Displot KDE":
        sns.displot(data=data, x=column_name, kind='kde')
    elif plot_type == "Displot Histogram":
        sns.displot(data=data, x=column_name, kind='hist')
    elif plot_type == "Displot ECDF":
        sns.displot(data=data, x=column_name, kind='ecdf')
    else:
        st.warning('Choose a plot type')
    plt.title(f"{plot_type} of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    
    # Save plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    
    # Return the image bytes
    plot_image =  buffer.getvalue()

        # Display the image in Streamlit
    st.image(plot_image, use_column_width=True)  # Adjust display as needed

    return plot_image
  
def bivariate_plot(plot_type, data, column1, column2):
    categorical_columns = data.select_dtypes(include=['object']).columns
    add_hue = False
    hue_column_description = ""
    if len(categorical_columns)>0:
        add_hue = st.checkbox("Add Hue (Optional)")
        hue_column_description = ""
        if add_hue:
            hue_column = st.selectbox("Select Hue Column:", categorical_columns, key='hue_column')
            hue_column_description = st.text_input(f"Enter the description of the {hue_column} Column to get better insights: ")
    if plot_type == "Scatter Plot":
        sns.scatterplot(data=data, x=column1, y=column2,
                        hue=hue_column if add_hue else None)
    elif plot_type == "Line Plot":
        sns.lineplot(data=data, x=column1, y=column2,
                     hue=hue_column if add_hue else None)
    elif plot_type == "Heatmap":
        numeric_data = data.select_dtypes(include='number')
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    elif plot_type == "Pair Plot":
        sns.pairplot(data, hue=hue_column if add_hue else None)
    elif plot_type == "Joint Plot":
        sns.jointplot(data=data, x=column1, y=column2,
                      hue=hue_column if add_hue else None)
    elif plot_type == "Violin Plot":
        sns.violinplot(data=data, x=column1, y=column2,
                       hue=hue_column if add_hue else None, split=True)
    elif plot_type == "Box Plot":
        sns.boxplot(data=data, x=column1, y=column2,
                    hue=hue_column if add_hue else None)
    elif plot_type == "Bar Plot":
        sns.barplot(data=data, x=column1, y=column2,
                    hue=hue_column if add_hue else None)
    # Clear the buffer and reset the file pointer to the beginning
    plt.title(f"{plot_type} between {column1} and {column2}")
    plt.xlabel(column1)
    plt.ylabel(column2)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()

    plot_image = buffer.getvalue()
    st.image(plot_image, use_column_width=True) 
    return plot_image, hue_column_description


def generate_univariate_prompt(column_name, column_description, plot_type):
    prompt = f"Give me insights from this plot. It is a univariate plot of type {plot_type} of the column: \"{column_name}\". And here is the description of the column: \"{column_description}\""
    return prompt

def generate_bivariate_prompt(column1, column2, column1_description, column2_description, hue_column_description, plot_type):
    prompt = f"Give me insights from this plot. It is a bivariate plot of type {plot_type} of the columns: {column1} and {column2}.Here is the description for column \"{column1}\": \"{column1_description}\" . And here is the description for column \"{column2}\": \"{column2_description}\" and here is the description for the hue column: \"{hue_column_description}\"."
    return prompt


instructions = """
Your Role:

Data Visualization Expert:
 You are a powerful assistant designed to analyze graphs and plots using the Gemini Pro Vision API.
Insights Generator:
 Your goal is to help users understand and extract meaningful insights from visual data.

Clarity & Depth:

Provide clear, concise, and informative explanations that are easy to understand, even for non-experts.
Avoid technical jargon and explain complex concepts in simple terms.
Go beyond surface observations and reveal hidden trends or interesting patterns in the data.

Structure & Readability:

Headings:
 Use clear and concise headings that accurately reflect the overall message of the graph.
Bullet Points:
 Organize your insights into a number bulleted list for improved readability with numbering 1,2,3....so on.
Minimum Bullet Points:
 Aim for at least 5-10 insightful bullet points per response, depending on the complexity of the graph.
Explanations with Nuance:
 Under each bullet point, provide detailed explanations that go beyond a single sentence.
Explanation Formatting:
 Use bold font for "Explanation:" and a new line for better separation.

Versatility & Grace:

Be adaptable to handle various types of graphs (bar charts, scatter plots, etc.)
Gracefully handle unexpected data formats or missing information.
Overall Objective:

Empowering Users:
Your primary goal is to be an invaluable assistant, helping users unlock the full potential of their visual data and gain actionable insights.
By incorporating these refinements, your model's responses will be clearer, more informative, and ultimately more helpful for users trying to understand their data.

"""
@st.cache_data(show_spinner=False)
def get_insights(instructions, prompt, plot_image):
    model = genai.GenerativeModel('gemini-pro-vision')
    plot_image = Image.open(io.BytesIO(plot_image))
    response = model.generate_content([instructions, prompt, plot_image])
    response = response.text
    return response

        

def main():
    st.markdown("<p style='{}'>➡️created by 'Muhammad Zain Attiq'</p>".format(created_style), unsafe_allow_html=True)
    st.markdown(f"<h1 style='{header_style}'>AI Powered Data Insights</h1>", unsafe_allow_html=True)
    data = pd.DataFrame()
    with st.expander("About the app..."):
        st.info("This is a AI powered complete data visualization toolkit web app which can automatically generate plots and graphs of your data features and also provides the insight into those plots and graphs. You just have to select the type of plot you want for your selected column. It will automatically do everything for")
    # upload_type =  st.selectbox("Data: ", ["Select", "Upload Data", "Use Sample Data", ], key='upload_data_type')
    data = upload_data()
    if st.checkbox("Use Sample Data"):
        data_type = st.selectbox("Data: ", ["Choose Data...", "titanic data", "tips"], key='sample_data_type')
        if data_type == 'titanic data':
            data = sns.load_dataset("titanic")
            st.write(data.head())
        elif data_type == 'tips':
            data = sns.load_dataset("tips")
            st.write(data.head())
    if not data.empty:
        data_description = st.text_input("Enter data description to get better insights:")
        vis_type_value = st.selectbox("Select visualization type", ["Select:",
                            "Univariate", "Bivariate"], key='vis_type_key')

        if vis_type_value == "Univariate":
            column_name = st.radio("Select column", list(data.columns), key='univariate_column')
            column_description = st.text_input(
        f"Enter description for column '{column_name}' to get better insights: ", key=column_name)
            plot_type = st.selectbox("Select Plot Type:", ["Select", "Histogram", "Kernel Density Estimate (KDE)",
                                            "Box Plot", "Rug Plot", "ECDF Plot",
                                            "Displot KDE", "Displot Histogram", "Displot ECDF"], key='univariate_plot_type')
            plot_image =  univariate_plot(plot_type, data, column_name)
            univariate_prompt = generate_univariate_prompt(column_name, column_description, plot_type)
            if st.button("Get Insights"):
                with st.spinner("Getting Insights..."):
                    response = get_insights(instructions, univariate_prompt, plot_image)
                    st.markdown('<h1 class="response_style">Response</h1>', unsafe_allow_html=True)
                    st.write(response)
        elif vis_type_value == "Bivariate":
            col1, col2 = st.columns(2)
            with col1:
                column1 = st.selectbox("Select column for x-axis", data.columns, key='bivariate_column1')
            with col2:
                column2 = st.selectbox("Select column for y-axis", data.columns, key='bivariate_column2')
            column1_description = st.text_input(
        f"Enter description for {column1} Column to get better insights::",  key=f'{column1}_description')
            column2_description = st.text_input(
        f"Enter description for {column2} Column to get better insights::")
            plot_type = st.selectbox("Select Plot Type:", [
                        "Scatter Plot", "Line Plot", "Heatmap", "Pair Plot", "Joint Plot", "Violin Plot", "Box Plot", "Bar Plot"], key='bivariate_plot_type')
            plot_image, hue_column_description =  bivariate_plot(plot_type, data, column1, column2)
            bivariate_prompt = generate_bivariate_prompt(column1, column2, column1_description, column2_description, hue_column_description, plot_type)
            if st.button("Get Insights"):
                with st.spinner("Getting Insigths..."):
                    response = get_insights(instructions, bivariate_prompt, plot_image)
                    st.header("Response")
                    st.markdown('<h1 class="response_style">Response</h1>', unsafe_allow_html=True)
        
        else:
            st.warning("Select one of these.")
if __name__ == '__main__':
    try:
      main()
    except Exception as e:
      st.error(f"An problem occurred: {e}")