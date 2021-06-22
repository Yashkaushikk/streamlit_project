from pandas._config.config import options
import streamlit as st
import pandas as pd


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()



st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename)

    return taxi_data


with header:
    st.title('Welcome to my First Streamlit Project')
    st.text('This is some text')

with dataset:
    st.header('Taxi dataset')
    st.text('This dataset will be easily available to you on ...site')

    taxi_data = get_data('data/taxi_data.csv')
    # st.write(taxi_data.head(5))

    st.subheader('Pick-up Location Distribution on this Dataset')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(40)
    st.bar_chart(pulocation_dist)

with features:
    st.header('This contains the features')

    st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic..')
    st.markdown('* **Second feature:** I created this feature because of this... I calculated it using this logic..')

with model_training:
    st.header('This will train the model')
    st.text('Here you can choose the hyper parameters and check how your model respond to changes')

    sel_col,disp_col = st.beta_columns(2)

    max_depth  = sel_col.slider('what should be the Max depth of the model?', min_value=10 , max_value =100,value = 20 , step = 10)
    n_estimators = sel_col.selectbox('How many trees should be there? ' , options=[100,200,300,'No limit'], index = 0)

    sel_col.text('List of features in my dataset:')
    sel_col.write(taxi_data.columns)

    input_feature = sel_col.text_input('Which feature should be there as an input feature' , 'PULocationID')


    if n_estimators == 'No limit':
	
    	 regr = RandomForestRegressor(max_depth= max_depth)
	
    else:
		 
         regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)


X = taxi_data[[input_feature]]
y = taxi_data[['trip_distance']]

regr.fit(X, y)
prediction = regr.predict(y)

disp_col.subheader('Mean absolute error of the model is:')
disp_col.write(mean_absolute_error(y, prediction))

disp_col.subheader('Mean squared error of the model is:')
disp_col.write(mean_squared_error(y, prediction))

disp_col.subheader('R squared score of the model is:')
disp_col.write(r2_score(y, prediction))




