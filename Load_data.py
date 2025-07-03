from Comparative_models.BBO_CFAT import BBO_CFAT_1   # 1st comparative model
from Comparative_models.CNN import CNN_1             # 2nd comparative model
from Comparative_models.SAE_CNN import SAE_CNN       # 3rd comparative model
from Comparative_models.CNN_LSTM import CNN_LSTM_1   # 4th comparative model
from Comparative_models.PM_WA import PM_WA_1         # 5th comparative model
from Comparative_models.RNN import RNN_1             # 6th comparative model
from Comparative_models.EAI import ELAI_1            # 7th comparative model  == all comparative models are imported from the Comparative models directory
import numpy as np


def Load_data2(data):   # function used for loading data when it is runned from the root directory

    feat=np.load(f"data_loader/{data}_features.npy")  # loading features
    labels=np.load(f"data_loader/{data}_labels.npy")  # loading labels
    return feat, labels

def Load_data(data):  # function used for loading data when it is called from some inside directory

    feat=np.load(f"../data_loader/{data}_features.npy")  # loading features
    labels=np.load(f"../data_loader/{data}_labels.npy")  # loading labels
    return feat, labels
def train_test_split2(balanced_feat,balanced_label,percent):

    data_size = balanced_feat.shape[0]  # Checks the shape of balanced_feat to convert the training_percentage to integer
    actual_percentage = int((data_size / 100) * percent)  # Converted the float training percentage to integer
    training_sequence = balanced_feat[:actual_percentage]  # splitting the training data
    training_labels = balanced_label[:actual_percentage]  # splitting the training label
    testing_sequence = balanced_feat[actual_percentage:]   # splitting the testing sequence
    testing_labels = balanced_label[actual_percentage:]    # splitting the Testing labels

    return training_sequence,testing_sequence,training_labels,testing_labels   #The function  train_test_split1 return the training and testing data


def models_return_metrics(data,epochs,ok=True,percent=None):  # this is the function inside which the comparative models are called ans Analysis is performed
    from Proposed_model.PM1 import proposed_model_main # importing proposed model inside the function as it creates an error circular import
    from Proposed_model.PM_test import proposed_model
    training_percentage = [40, 50, 60, 70, 80, 90]  # training percentage when the models iterates over it

    if ok:
        # Initialize empty lists for each model
        BBO_CFAT_metrics_all = []
        CNN_metrics_all = []
        CNN_LSTM_metrics_all = []
        EAI_metrics_all = []
        PM_WA_metrics_all = []
        RNN_metrics_all = []
        SAE_CNN_metrics_all = []

        for i in training_percentage:
            print(f"The training for comparative model with {i}% training is starting")
            feat1, label1 = Load_data2(data)  # loading data
            feat, label = balance2(data, feat1, label1) # balancing it
            x_train, x_test, y_train, y_test = train_test_split2(feat,label, percent=i)  # splitting it into train and test

            BBO_CFAT_metrics_all.append(BBO_CFAT_1(x_train, x_test, y_train, y_test, epochs)) # data is passed to model one and metrics are returned which will be appended to the empty list (same procedure for all models below)

            CNN_metrics_all.append(CNN_1(x_train, x_test, y_train, y_test, epochs))

            CNN_LSTM_metrics_all.append(CNN_LSTM_1(x_train, x_test, y_train, y_test, epochs))

            EAI_metrics_all.append(ELAI_1(x_train, x_test, y_train, y_test, epochs))

            PM_WA_metrics_all.append(PM_WA_1(x_train, x_test, y_train, y_test, epochs))

            RNN_metrics_all.append(RNN_1(x_train, x_test, y_train, y_test, epochs))

            SAE_CNN_metrics_all.append(SAE_CNN(x_train, x_test, y_train, y_test, epochs))


        return (BBO_CFAT_metrics_all, CNN_metrics_all,CNN_LSTM_metrics_all,
                EAI_metrics_all, PM_WA_metrics_all,RNN_metrics_all,SAE_CNN_metrics_all) # returning all the metrics

    else:

        feat1,label1=Load_data2(data)
        feat, label = balance2(data,feat1,label1)
        x_train, x_test, y_train, y_test = train_test_split2(feat, label, percent)

        BBO_metrics1 = BBO_CFAT_1(x_train, x_test, y_train, y_test, epochs=epochs)

        CNN_metrics1 = CNN_1(x_train, x_test, y_train, y_test, epochs)

        CNN_LSTM_metrics1 = CNN_LSTM_1(x_train, x_test, y_train, y_test, epochs)

        ELAI_metrics1 = ELAI_1(x_train, x_test, y_train, y_test, epochs)

        PM_WA_metrics1 = PM_WA_1(x_train, x_test, y_train, y_test, epochs)

        RNN_metrics1 = RNN_1(x_train, x_test, y_train, y_test, epochs)

        SAE_CNN_metrics1 = SAE_CNN(x_train, x_test, y_train, y_test, epochs)

        PM_metrics_1 = proposed_model(x_train, x_test, y_train, y_test, epochs,data)

        return (BBO_metrics1,CNN_metrics1,CNN_LSTM_metrics1,
                ELAI_metrics1,PM_WA_metrics1,RNN_metrics1,SAE_CNN_metrics1,PM_metrics_1)


def balance(DB,full_dataset, label_col=None,labels=None):
    balanced_feat=None
    balanced_lab=None
    if DB=="UNSW-NB15":
        label = full_dataset[label_col]
        # We are doing the below steps to take equal amount of data from all the classes
        class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
        class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
        class_2_indices = np.where(label == 2)[0]  # this line chooses all the 2-label index in the label data
        class_3_indices = np.where(label == 3)[0]  # this line chooses all the 3-label index in the label data
        class_4_indices = np.where(label == 4)[0]  # this line chooses all the 4-label index in the label data
        class_5_indices = np.where(label == 5)[0]  # this line chooses all the 5-label index in the label data
        class_6_indices = np.where(label == 6)[0]  # this line chooses all the 6-label index in the label data
        class_7_indices = np.where(label == 7)[0]  # this line chooses all the 7-label index in the label data
        class_8_indices = np.where(label == 8)[0]  # this line chooses all the 8-label index in the label data
        class_9_indices = np.where(label == 9)[0]  # this line chooses all the 9-label index in the label data
        class_10_indices = np.where(label == 10)[0]  # this line chooses all the 10-label index in the label data
        class_11_indices = np.where(label == 11)[0]  # this line chooses all the 11-label index in the label data
        class_12_indices = np.where(label == 12)[0]  # this line chooses all the 12-label index in the label data
        class_13_indices = np.where(label == 13)[0]  # this line chooses all the 12-label index in the label data



        selected_class_0 = np.random.choice(class_0_indices, 5050,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_1 = np.random.choice(class_1_indices, 10000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_2 = np.random.choice(class_2_indices, 10000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_3 = np.random.choice(class_3_indices, 1200,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_4 = np.random.choice(class_4_indices, 2600,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_5 = np.random.choice(class_5_indices, 1500,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_6 = np.random.choice(class_6_indices, 500,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_7 = np.random.choice(class_7_indices, 10000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_8 = np.random.choice(class_8_indices, 10000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_9 = np.random.choice(class_9_indices, 20000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_10 = np.random.choice(class_10_indices, 1500,
                                             replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_11 = np.random.choice(class_11_indices, 223,
                                             replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_12 = np.random.choice(class_12_indices, 174,
                                             replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_13 = np.random.choice(class_13_indices, 10000,
                                             replace=False)  # randomly chooses 1000 indices from the class_0_indices

        selected_indices = np.concatenate(
            [selected_class_0, selected_class_1, selected_class_2, selected_class_3, selected_class_4, selected_class_5,
             selected_class_6, selected_class_7, selected_class_8, selected_class_9, selected_class_10,
             selected_class_11,selected_class_12, selected_class_13])  # joining all the classes together
        np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

        balanced_feat = full_dataset.iloc[selected_indices]
    if DB=="N-BaIoT":
        label = full_dataset[label_col]
        # We are doing the below steps to take equal amount of data from all the classes
        class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
        class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
        class_2_indices = np.where(label == 2)[0]  # this line chooses all the 2-label index in the label data
        class_3_indices = np.where(label == 3)[0]  # this line chooses all the 3-label index in the label data
        class_4_indices = np.where(label == 4)[0]  # this line chooses all the 4-label index in the label data



        selected_class_0 = np.random.choice(class_0_indices, 30000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_1 = np.random.choice(class_1_indices, 30000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_2 = np.random.choice(class_2_indices, 477,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_3 = np.random.choice(class_3_indices, 30000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_4 = np.random.choice(class_4_indices, 79,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices

        selected_indices = np.concatenate(
            [selected_class_0, selected_class_1, selected_class_2, selected_class_3,
             selected_class_4])  # joining all the classes together
        np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

        balanced_feat = full_dataset.iloc[selected_indices]

    if DB=="CICIDS2015":
        label = labels
        # We are doing the below steps to take equal amount of data from all the classes
        class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
        class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
        class_2_indices = np.where(label == 2)[0]  # this line chooses all the 2-label index in the label data
        class_3_indices = np.where(label == 3)[0]  # this line chooses all the 3-label index in the label data
        class_4_indices = np.where(label == 4)[0]  # this line chooses all the 4-label index in the label data
        class_5_indices = np.where(label == 5)[0]  # this line chooses all the 5-label index in the label data
        class_6_indices = np.where(label == 6)[0]  # this line chooses all the 6-label index in the label data
        class_7_indices = np.where(label == 7)[0]  # this line chooses all the 7-label index in the label data
        class_8_indices = np.where(label == 8)[0]  # this line chooses all the 8-label index in the label data
        class_9_indices = np.where(label == 9)[0]  # this line chooses all the 9-label index in the label data



        selected_class_0 = np.random.choice(class_0_indices, 20000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_1 = np.random.choice(class_1_indices, 385,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_2 = np.random.choice(class_2_indices, 452,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_3 = np.random.choice(class_3_indices, 4467,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_4 = np.random.choice(class_4_indices, 20000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_5 = np.random.choice(class_5_indices, 20000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_6 = np.random.choice(class_6_indices, 4467,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_7 = np.random.choice(class_7_indices, 16735,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_8 = np.random.choice(class_8_indices, 2102,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_9 = np.random.choice(class_9_indices, 246,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_indices = np.concatenate(
            [selected_class_0, selected_class_1, selected_class_2, selected_class_3, selected_class_4, selected_class_5,
             selected_class_6, selected_class_7, selected_class_8,
             selected_class_9])  # joining all the classes together
        np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

        balanced_feat = full_dataset.iloc[selected_indices]
        balanced_lab = labels.loc[selected_indices]
    if DB=="CICIDS2015":
        return balanced_feat,balanced_lab
    else:
        return balanced_feat



def balance2(DB,feat,labels):
    balanced_feat=None
    balanced_lab=None
    if DB=="UNSW-NB15":
        label = labels
        # We are doing the below steps to take equal amount of data from all the classes
        class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
        class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
        class_2_indices = np.where(label == 2)[0]  # this line chooses all the 0-label index in the label data
        class_3_indices = np.where(label == 3)[0]  # this line chooses all the 1-label index in the label data
        class_4_indices = np.where(label == 4)[0]  # this line chooses all the 0-label index in the label data
        class_5_indices = np.where(label == 5)[0]  # this line chooses all the 1-label index in the label data
        class_6_indices = np.where(label == 6)[0]  # this line chooses all the 0-label index in the label data
        class_7_indices = np.where(label == 7)[0]  # this line chooses all the 1-label index in the label data
        class_8_indices = np.where(label == 8)[0]  # this line chooses all the 0-label index in the label data
        class_9_indices = np.where(label == 9)[0]  # this line chooses all the 1-label index in the label data
        class_10_indices = np.where(label == 10)[0]  # this line chooses all the 0-label index in the label data
        class_11_indices = np.where(label == 11)[0]  # this line chooses all the 1-label index in the label data
        class_12_indices = np.where(label == 12)[0]  # this line chooses all the 0-label index in the label data
        class_13_indices = np.where(label == 13)[0]  # this line chooses all the 0-label index in the label data




        selected_class_0 = np.random.choice(class_0_indices, 200,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_1 = np.random.choice(class_1_indices, 200,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_2 = np.random.choice(class_2_indices, 200,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_3 = np.random.choice(class_3_indices, 200,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_4 = np.random.choice(class_4_indices, 200,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_5 = np.random.choice(class_5_indices, 200,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_6 = np.random.choice(class_6_indices, 200,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_7 = np.random.choice(class_7_indices, 200,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_8 = np.random.choice(class_8_indices, 200,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_9 = np.random.choice(class_9_indices, 200,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_10 = np.random.choice(class_10_indices, 200,
                                             replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_11 = np.random.choice(class_11_indices, 200,
                                             replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_12 = np.random.choice(class_12_indices, 200,
                                             replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_13 = np.random.choice(class_13_indices, 200,
                                             replace=False)  # randomly chooses 1000 indices from the class_0_indices

        selected_indices = np.concatenate(
            [selected_class_0, selected_class_1, selected_class_2, selected_class_3, selected_class_4, selected_class_5,
             selected_class_6, selected_class_7, selected_class_8, selected_class_9, selected_class_10,
             selected_class_11,selected_class_12, selected_class_13])  # joining all the classes together
        np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

        balanced_feat = feat[selected_indices]
        balanced_lab=labels[selected_indices]




    if DB=="N-BaIoT":
        label = labels
        # We are doing the below steps to take equal amount of data from all the classes
        class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
        class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
        class_2_indices = np.where(label == 2)[0]  # this line chooses all the 0-label index in the label data
        class_3_indices = np.where(label == 3)[0]  # this line chooses all the 1-label index in the label data
        class_4_indices = np.where(label == 4)[0]  # this line chooses all the 0-label index in the label data



        selected_class_0 = np.random.choice(class_0_indices, 300,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_1 = np.random.choice(class_1_indices, 300,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_2 = np.random.choice(class_2_indices, 300,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_3 = np.random.choice(class_3_indices, 300,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_4 = np.random.choice(class_4_indices, 300,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices

        selected_indices = np.concatenate(
            [selected_class_0, selected_class_1, selected_class_2, selected_class_3,
             selected_class_4])  # joining all the classes together
        np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

        balanced_feat = feat[selected_indices]
        balanced_lab=labels[selected_indices]


    if DB=="CICIDS2015":
        label = labels
        # We are doing the below steps to take equal amount of data from all the classes
        class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
        class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
        class_2_indices = np.where(label == 2)[0]  # this line chooses all the 0-label index in the label data
        class_3_indices = np.where(label == 3)[0]  # this line chooses all the 1-label index in the label data
        class_4_indices = np.where(label == 4)[0]  # this line chooses all the 0-label index in the label data
        class_5_indices = np.where(label == 5)[0]  # this line chooses all the 1-label index in the label data
        class_6_indices = np.where(label == 6)[0]  # this line chooses all the 0-label index in the label data
        class_7_indices = np.where(label == 7)[0]  # this line chooses all the 1-label index in the label data
        class_8_indices = np.where(label == 8)[0]  # this line chooses all the 0-label index in the label data
        class_9_indices = np.where(label == 9)[0]  # this line chooses all the 1-label index in the label data



        selected_class_0 = np.random.choice(class_0_indices, 200,
                                            replace=False)
        selected_class_1 = np.random.choice(class_1_indices, 200,
                                            replace=False)

        selected_class_2 = np.random.choice(class_2_indices, 200,
                                            replace=False)
        selected_class_3 = np.random.choice(class_3_indices, 200,
                                            replace=False)
        selected_class_4 = np.random.choice(class_4_indices, 200,
                                            replace=False)
        selected_class_5 = np.random.choice(class_5_indices, 200,
                                            replace=False)
        selected_class_6 = np.random.choice(class_6_indices, 200,
                                            replace=False)
        selected_class_7 = np.random.choice(class_7_indices, 200,
                                            replace=False)
        selected_class_8 = np.random.choice(class_8_indices, 200,
                                            replace=False)
        selected_class_9 = np.random.choice(class_9_indices, 200,
                                            replace=False)

        selected_indices = np.concatenate(
            [selected_class_0, selected_class_1, selected_class_2, selected_class_3, selected_class_4, selected_class_5,
             selected_class_6, selected_class_7, selected_class_8,
             selected_class_9])  # joining all the classes together
        np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

        balanced_feat = feat[selected_indices]
        balanced_lab = labels[selected_indices]


    return balanced_feat,balanced_lab