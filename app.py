import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

# Load the trained machine learning model
model = pickle.load(open('RandomForestClassifi.pk1' , 'rb'))

# Define a function to make predictions
def predict(inputs):
    # Preprocess the inputs if needed
    # Make predictions
    output = model.predict(inputs.values.reshape(1, -1))[0]
    return output

def get_proba(inputs):
    output = model.predict_proba(inputs.values.reshape(1, -1))[0]
    return output

p_name = ["ph(0-14)","Hardness(0 mg/l - 400 mg/l)","Solids(0 mg/l - 65000 mg/l)","Chloramines(0 mg/l - 14 mg/l)","Sulfate(0 mg/l - 500 mg/l)"
    ,"Conductivity(0 μS/cm - 800 μS/cm)","Organic_carbon(0 mg/l - 30 mg/l)","Trihalomethanes(0 mg/l - 120 mg/l)","Turbidity(0 NTU - 7 NTU)"]

# Define the Streamlit app
def main():
    # Set the title and description of your app
    st.title('Water Potabilty Prediction.')
    st.write('Enter the numerical inputs below:') 

    # Create input fields for 9 numerical inputs
    inputs = []
    for i in range(9):
        inputs.append(st.number_input(f'{p_name[i]}', min_value=0.0))

    # Create a button to trigger the prediction
    if st.button('Predict'):
        inputs = pd.Series(inputs)
        # Validate inputs if needed

        # Call the predict function
        output = predict(inputs)
        probability = get_proba(inputs)
        if output == 0.0:
            # st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)
            st.write(f'Predicted Output: The water with above parameters is not potable. (Probability : {probability[0]*100}%)')
        else:
            st.write(f'Predicted Output: The water with above parameters is potable. (Probability : {probability[0]*100}%)')
        # st.write(f'Predicted Output: {output}')

if __name__ == '__main__':
    main()
