# write in predict_page.py ya 3aak in vs code

import streamlit as st
import pickle  
import numpy as np

#from Lungs_Disease_Classification_ML_DL import rfst
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV



#rfst()
# def load_model():
#     with open('Predict_model.pkl', 'rb') as file:
#         data = pickle.load(file)  
#         rfst_loaded=data["model"]  
#     return rfst_loaded

# Mode=load_model()
df = pd.read_csv("cancer patient data sets.csv", sep=",",encoding="UTF-8")
df6 = pd.read_csv("cancer patient data sets.csv", sep=",",encoding="UTF-8")
x = df6.drop(["Level","Patient Id","index"],axis=1).values
y = df.Level.values
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# rfst = RandomForestClassifier(n_estimators=10,random_state=42)
# rfst.fit(x_train_scaled, y_train)
# y_pred_rfst = rfst.predict(x_test_scaled)

lr = LogisticRegression()
lr.fit(x_train , y_train)
lr.score(x_train , y_train)
lr.score(x_test , y_test)
y_pred=lr.predict(x_test)
data=pd.DataFrame({'y_Test  ':y_test,'y_pred  ':y_pred})
data[:20]

filename= 'Predict_model.pkl'
pickle.dump(lr,open(filename,'wb'))
Mode= pickle.load(open('Predict_model.pkl', 'rb'))


def show_predict_page():
    st.title("""Lungs Cancer Recognition""")
    st.write("""We need some information to recognize the data""")
    st.write("""So please fill this form""")

    age=st.number_input("Age",step=1.,format="%.6f")
    st.write('The current Age is ', age)
    gender=st.number_input("Gender",step=1.,format="%.6f")
    st.write('Gender is ', gender)
    airPollution=st.number_input("Air Pollution",step=1.,format="%.6f")
    st.write('The Air Pollution is ', airPollution)
    alcoholUse=st.number_input("Alcohol use",step=1.,format="%.6f")
    st.write('The Alcohol use is ', alcoholUse)
    dustAllergy=st.number_input("Dust Allergy",step=1.,format="%.6f")
    st.write('The Dust Allergy is ', dustAllergy)
    occuPationalHazards=st.number_input("OccuPational Hazards",step=1.,format="%.6f")
    st.write('The OccuPational Hazards is ', occuPationalHazards)
    geneticRisk=st.number_input("Genetic Risk",step=1.,format="%.6f")
    st.write('The Genetic Risk is ', geneticRisk)
    chronicLungDisease=st.number_input("chronic Lung Disease",step=1.,format="%.6f")
    st.write('The chronic Lung Disease is ', chronicLungDisease)
    balancedDiet=st.number_input("Balanced Diet",step=1.,format="%.6f")
    st.write('The Balanced Diet is ', balancedDiet)
    obesity=st.number_input("Obesity",step=1.,format="%.6f")
    st.write('The Obesity is ', obesity)       
    smoking=st.number_input("Smoking",step=1.,format="%.6f")
    st.write('The Smoking is ', smoking)
    passiveSmoker=st.number_input("Passive Smoker",step=1.,format="%.6f")
    st.write('The Passive Smoker is ', passiveSmoker)
    chestPain=st.number_input("Chest Pain",step=1.,format="%.6f")
    st.write('The Chest Pain is ', chestPain)
    coughingOfBlood=st.number_input("Coughing of Blood",step=1.,format="%.6f")
    st.write('The Coughing of Blood is ', coughingOfBlood)
    fatigue=st.number_input("Fatigue",step=1.,format="%.6f")
    st.write('The Fatigue is ', fatigue)
    weightLoss=st.number_input("Weight Loss",step=1.,format="%.6f")
    st.write('The Weight Loss is ', weightLoss)
    shortnessOfBreath=st.number_input("Shortness of Breath",step=1.,format="%.6f")
    st.write('The Shortness of Breath is ', shortnessOfBreath)
    wheezing=st.number_input("Wheezing",step=1.,format="%.6f")
    st.write('The Wheezing is ', wheezing)
    swallowingDifficulty=st.number_input("Swallowing Difficulty",step=1.,format="%.6f")
    st.write('The Swallowing Difficulty is ', swallowingDifficulty)
    clubbingOfFingerNails=st.number_input("Clubbing of Finger Nails",step=1.,format="%.6f")
    st.write('The Clubbing of Finger Nails is ', clubbingOfFingerNails)
    frequentCold=st.number_input("Frequent Cold",step=1.,format="%.6f")
    st.write('The Frequent Cold is ', frequentCold)
    dryCough=st.number_input("Dry Cough",step=1.,format="%.6f")
    st.write('The Dry Cough is ', dryCough)
    snoring=st.number_input("Snoring",step=1.,format="%.6f")
    st.write('The Snoring is ', snoring)

    ok=st.button("Predict")
    if ok:
        X=[[age,gender,airPollution,alcoholUse,dustAllergy,occuPationalHazards,geneticRisk,chronicLungDisease,balancedDiet,obesity,smoking,passiveSmoker,chestPain,coughingOfBlood,fatigue,weightLoss,shortnessOfBreath,wheezing,swallowingDifficulty,clubbingOfFingerNails,frequentCold,dryCough,snoring]]
        # mlp_loaded = data["model"]
        Prediction=Mode.predict(X)



        st.subheader(f"Prediction of the disease is {Prediction[0]}")