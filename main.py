import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import export_graphviz
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv('symptoms.csv')
testing  = pd.read_csv('medicine.csv')
cols     = training.columns
cols     = cols[:-1]
x        = training[cols]
y        = training['prognosis']
y1       = y

reduced_data = training.groupby(training['prognosis']).max()

list1 = ['Chlotrimazole','benadryl','maalox','obeticholic','benadryl','amoxicillin','metaformin','abacabir','Ib-profin,albuterol',
'angiotensin converting enzyme','limitrex','naproxensodium,''labetalol','hyperbilirubyninia','malarone','acitamenophene','acitamenophene',
'polymixin','no medicine specifically just rest','entecabar','harvoni','pegylated interferonalpha','ribavirin','pentoxifylline','isoniazid',
'paracetamol','ciproflaxacin','hydrocortison','Nitroglycerin : ipbprofin and many','self care','levoxyl','tapazole','glucagon',
'tylenol and other acitaminophen','tiramcinolone','epleymaneuber','ratnoids as well as dapsone','bactrim, septra','enbrel remicade','altabax'
]

list2 = ['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction','Peptic ulcer diseae','AIDS','Diabetes ','Gastroenteritis',
'Bronchial Asthma','Hypertension ','Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue',
'Typhoid','hepatitis A,''Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia',
'Dimorphic hemmorhoids(piles)','Heart attack','Varicose veins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis','Arthritis',
'(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis','Impetigo'
]

#print(len(list1))
#print(len(list2))

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']
testy    = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)


importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


print("--------------------ReAsOnInG SyStEm------------------")
print("\nPlease reply Yes or No for the following symptoms")
def print_disease(node):
    #print(node)
    node = node[0]
    #print(len(node))
    val  = node.nonzero()
    #print(val)
    disease = le.inverse_transform(val[0])
    return disease
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    #print(tree_)
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print("def tree({}):".format(", ".join(feature_names)))
    symptoms_present = []
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("\nAnswer to the following questions ==> ")
            print(name + " ?")
            ans = input()
            ans = ans.lower()
            if ans == 'yes':
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            print("   \n              EVALUATION                 ")
            print("\nOn the basis of your evaluation ==>")
            print("------------------------------------------------------")
            print( "You may have :" +  present_disease )
            for i in range(len(list2)):
                if list2[i] == present_disease:
                    print("\nMedication required : ", list1[i])



            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("\nsymptoms present : " + str(list(symptoms_present)))
            print("------------------- + ---------------------------------")
    recurse(0, 1)

tree_to_code(clf,cols)
