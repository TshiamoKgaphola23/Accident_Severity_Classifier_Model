import streamlit as st
import pandas as pd
import pickle

# Load the model
model_file = 'accident_severity_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Load the data
# data_file = 'acc_veh_dataset.csv'
# data = pd.read_csv(data_file)

# Define label mapping
label_mapping = {
    '0': 'Fatal',
    '1': 'Serious',
    '2': 'Slight'
}

# Background
background_image_url = "https://img.pikbest.com/ai/illus_our/20230424/f0a5032938aeaecad6e2c9d5e3dca32a.jpg!w700wp"
background_css = """
<style>
    .stApp {{
        background: url("{image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
</style>
""".format(image=background_image_url)
st.markdown(background_css, unsafe_allow_html=True)

# Define features and target
features = ['1st_Road_Number', '2nd_Road_Number', 'Date', 
            'Local_Authority_(District)_x', 'Local_Authority_(Highway)_x', 
            'Longitude', 'Latitude',
            'LSOA_of_Accident_Location_x']
target = 'Accident_Severity_x'

# Main function to run the Streamlit app
def main():
    st.title('Accident Severity Classifier')
    st.sidebar.title('Accident Severity Inputs')

    # Display user input fields
    user_input = {}
    for feature in features:
        user_input[feature] = st.sidebar.text_input(feature, '')

    # Convert user input into DataFrame
    input_df = pd.DataFrame([user_input])

    if st.sidebar.button('Classify'):
        # Perform prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        # Map numeric result to label
        predicted_label = label_mapping.get(str(prediction[0]), 'Unknown')
        
        # Display prediction
        # st.subheader('Classification')
        st.write(f'**Predicted Severity**: {predicted_label}')

        # Display probability distribution with bold and percentage formatting
        st.write('**Probability Distribution:**')
        for i, label in label_mapping.items():
            percentage = probability[0][int(i)] * 100
            st.write(f'**{label}**: {percentage:.2f}%')

# Run the app
if __name__ == '__main__':
    main()
