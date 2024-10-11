from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import torch
from sklearn import preprocessing
from npeet import entropy_estimators as ee


def custom_CS(entropy_Y, entropy_labels_given_concepts, entropy_labels_given_inputs, CE_CONCEPTS):
    print(f'Entropy given concepts: {entropy_labels_given_concepts}, Entropy given inputs: {entropy_labels_given_inputs}')
    CS_num = entropy_Y - entropy_labels_given_concepts
    CS_den = entropy_Y - entropy_labels_given_inputs
    worst_case = CS_num / (CS_den + CE_CONCEPTS)
    
    return CS_num / CS_den, worst_case


def MI(X, Y):
    """
    Mutual Information (MI).
    """
    # Reshape the X data to be 2D from (a,b,c,...,z) to (n_samples,n_features)
    n_samples = X.shape[0]
    X = X.reshape(n_samples,-1)
    lab_enc = preprocessing.LabelEncoder()
    encoded = lab_enc.fit_transform(X)
    X = encoded
    print(X.shape, Y.shape)

    mi = mutual_info_classif(X[:,:], Y[:], discrete_features=False)
    indexed_numbers = list(enumerate(mi))
    sorted_indexed_numbers = sorted(indexed_numbers, key=lambda x: x[1], reverse=True)
    sorted_indexes = [index for index, value in sorted_indexed_numbers]
    sorted_values = [value for index, value in sorted_indexed_numbers]
    print('Sorted indexes', sorted_indexes)
    print('Sorted values', sorted_values)
    #print(ee.mi(X[:10000,20], Y[:10000]))
    #print(ee.mi(X[:10000,41], Y[:10000]))
    #print(ee.mi(X[:10000,:], Y[:10000]))
    
    return mutual_info_classif(X[:,:], Y[:], discrete_features=True)

def completeness_mig():
    """
    The Mutual Information Gap (MIG) is a measure of the disentanglement of the latent space.
    """
    pass

def completeness_modularity():
    """
    Modularity is a measure of the disentanglement of the latent space.
    """
    pass

def completeness_havasi_mi(labels, concepts, inputs):
    """
    Completeness score based on the Havasi method but using Mutual Information (MI) instead of the approximation described in the paper.
    """
    Y = labels
    C = concepts
    X = inputs
    #print(Y.shape)
    #print(C.shape)
    #print(X.shape)
    Y = Y.reshape(Y.shape[0], 1).astype(int)
    X = X.reshape(X.shape[0], -1)
    #Y = Y.tolist()
    #C = C.tolist()
    #X = X.tolist()
    #print(X.shape)
    #print(Y)
    #print(C)
    #print(X)
    #print(len(C))
    
    limit = 1000
    # Use the npeet entropy_estimators to calculate the MI
    # Using micd instead of mi 
    # The function micd allows a continuous (X or C) and a discrete variable Y 
    n = []
    d = []
    for i in range(30):
        numerator = ee.micd(C[0 +i*1000:limit +i*1000], Y[0+i*1000:limit+i*1000])
        #print('Numerator:', numerator)
        n.append(numerator)
        denominator = ee.micd(X[0+i*1000:limit+i*1000], Y[0+i*1000:limit+i*1000])
        #print('Denominator:', denominator)
        d.append(denominator)
        break

    numerator = sum(n) / len(n)
    denominator = sum(d) / len(d)
    micd = numerator / denominator

    # Get the boolean version of concepts
    C = (C > 0.5).astype(int)
    n = []
    for i in range(30):
        numerator = ee.midd(C[0 +i*1000:limit +i*1000], Y[0+i*1000:limit+i*1000])
        #print('Numerator:', numerator)
        n.append(numerator)
        break


    numerator = sum(n) / len(n)
    midd = numerator / denominator
    return micd, midd

def completeness_yeh():
    pass
    