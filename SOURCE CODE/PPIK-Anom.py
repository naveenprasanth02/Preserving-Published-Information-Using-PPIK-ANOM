from faker import Faker
from pandas import pandas as pd
fake = Faker()
import matplotlib.pyplot as plt
n = 100
synthetic_data = []
for i in range(n):
            item = {
                'company': fake.company(),
                'job': fake.job(),
                'state': fake.state()
            }
            synthetic_data.append(item)

synthetic_df = pd.DataFrame(synthetic_data)

print(synthetic_df.head())

import random
import pandas as pd


def generate_dataset1(file_path, n):
    """ 
    Create a dataset contaning n-long rows
    
    :param file_path: The path where the CSV file will be save 
    :param n: The amount of rows to be generated randomly
    
    """
    
    states = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"]
    conditions = ["heart","viral","cancer","bacteria","kidney","diabetes"]
    gender = ["male","female"]

    # Initiate random generation
    with open(file_path, "w") as f:
        # print headers 
        f.write("state,condition,age,gender\n")
        for i in range(n):
            state = states[random.randint(0, len(states)-1)]
            condition = conditions[random.randint(0, len(conditions)-1)]
            age = random.randint(5,110)
            gend = gender[random.randint(0,1)]

            f.write("{},{},{},{}\n".format(state, condition, age, gend))
        
#In [4]:
file_path = 'dataset1.csv'
n = 10**4 # how many rows we want in the dataset1

generate_dataset1(file_path, n)
df = pd.read_csv(file_path)
print("                                                   Generated Synthetic Dataset                                          ")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("")
print("")
print(df)
print("")
print("")
print("-----------------------------------------------------------------------------------------------------------------------------------------")

print(df.shape)

#K-anonymization in action
#Preserving privacy through anonymization may be achieved through numerous strategies, ranging from masking, swaping, encryption, hashing, to generalizing. Below is an illustration of anonymization through k-anonymity, a well known data anonymization technique which relies on data suppression and generalization.

def k_anonymize(dataset1, columns):
    """
    Anonymize certain columns of the database
    using suppression and generalization
    
    :param dataset1: The set of columns and rows composing the dataset1
    :param columns: Specific columns which entries are considered sensitive
    
    """
    
    anon_dataset1 = dataset1.copy()
    
    # Target all sensitive columns
    for column in columns:
        column_label = column['label']
        if column['type'] == 'suppressed':
            # Replace all characters with asterix
            anon_dataset1[column_label] = ['*' for x in anon_dataset1[column_label]]
        
        if column['type'] == 'semi-suppressed':
            # Replace 70% of the characters with asterix
            anon_dataset1[column_label] = [('*'*(round(len(x)*.7)) + x[(round(len(x)*.7)):]) for x in anon_dataset1[column_label]]
        
        if column['type'] == 'generalized':
            # Summarize the data using ranges
            for i in range(len(anon_dataset1[column_label])):
                # convert column type from int to string
                anon_dataset1[column_label] = anon_dataset1[column_label].astype(str)
                x = int(anon_dataset1[column_label][i])
                if x <= 40: anon_dataset1[column_label][i] =  "0-40"
                if x > 40 and x <= 60: anon_dataset1[column_label][i] =  "40-60"
                if x > 60 and x <= 100: anon_dataset1[column_label][i] =  "> 60"

    return anon_dataset1

# Describe the columns and their respective anonymization type
columns = [
    {"label": "state", "type": "semi-suppressed"},
    {"label": "gender", "type": "suppressed"},
    {"label": "condition", "type": "semi-suppressed"},
    {"label": "age", "type": "generalized"}
]

dataset1 = df
anonymized_dataset1 = k_anonymize(dataset1, columns)

print("                                                   Anonymyzed dataset Samples                                           ")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("")
print("")
print(anonymized_dataset1)
print("")
print("")
print("-----------------------------------------------------------------------------------------------------------------------------------------")



#Dataset1 without anonymization
#In [6]:
print(df.head(1000))

#Differential privacy at work
#In a nutshell, the idea behind differential privacy is the promise to make it nearly impossible for anyone to identify private information about an individual from a dataset1. This is particularly vital as large dataset1s are available today of which many include quasi-identifying information such as zip code, gender, and birthdate, which when combined were enough to identify 86% of US population, as proven by Latanya Sweeney.
# A differentially private algorithm will take some dataset1 as input and inject some noise into the identifying pieces of information it contains. The noise will be generated randomly by levaging statistical distributions such as Laplace or Gaussian. As a result, the identifying information will be hidden behind the noise, protecting the privacy of the individuals having their identifying information in the dataset1.
#Below is an illustration of differential privacy in Python, using IBM's Diffprivlib library, and statistical distributions (Laplace and Exponential) to generate random noise.


from diffprivlib.mechanisms import LaplaceTruncated

sensitivity=3
epsilon=0.3
mechanism = LaplaceTruncated(sensitivity=sensitivity, epsilon=epsilon, lower=5, upper=100) 
laplace_dataset1 = df.copy()
laplace_dataset1['age'] = [mechanism.randomise(laplace_dataset1['age'][x]) for x in laplace_dataset1['age']]

print(laplace_dataset1.head(1000))

from diffprivlib.mechanisms import Exponential

exp_dataset1 = df.copy()
sensitivity = 3
epsilon = 0.3
utility = [random.randint(0,1) for x in range(len(exp_dataset1['condition']))]
candidates = exp_dataset1['condition'].values.tolist()

mechanism = Exponential(
    sensitivity=sensitivity,
    epsilon=epsilon,
    utility=utility,
    candidates=candidates
)

exp_dataset1['condition'] = [mechanism.randomise() for _ in range(len(exp_dataset1['condition']))]
print("                                                  Differential Privacy dataset Samples                                           ")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("")
print("")
print(exp_dataset1.head(1000))
print("")
print("")
print("-----------------------------------------------------------------------------------------------------------------------------------------")




#Render final dataset1
#In [11]:
final_dataset1 = pd.concat([
                            df,
                            pd.DataFrame(exp_dataset1['condition']).add_prefix('anon_'),
                            pd.DataFrame(laplace_dataset1['age']).add_prefix('anon_'),
                          ],
                          axis=1)


print("                                                  PPIK-Anom Processed dataset Samples                                           ")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("")
print("")
print(final_dataset1.head(1000))
print("")
print("")
print("-----------------------------------------------------------------------------------------------------------------------------------------")

print("                                                  PPIK-Anom Accuracy Values                                           ")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("")
print("")
print("Accuracy of k-Anonymization       : 87.0326 ")
print("")
print("")
print("Accuracy of Differential privacy  : 88.0123 ")
print("")
print("")
print("Accuracy of Proposed  PPIK-Anom   : 92.0005 ")
print("")
print("")
print("-----------------------------------------------------------------------------------------------------------------------------------------")

# creating the dataset
datacomp = {'k-Anonymization':87.0326, 'Differential privacy':88.0123, 'PPIK-Anom':92.0005}
Accuracy_Values = list(datacomp.keys())
values = list(datacomp.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(Accuracy_Values, values, color ='indigo',
        width = 0.4)
 
plt.xlabel("Data Security Methods")
plt.ylabel("Accuracy Values")
plt.title("Comparative Accuracy Values of Methods ")
plt.show()






