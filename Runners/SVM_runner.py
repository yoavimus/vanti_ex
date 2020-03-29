import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix



class SVM_Runner:
    def __init__(self, logger,path):
        self.logger = copy.copy(logger)
        self.data_path = path


    def lin_svm(self):
        try:
            self.logger.info("start run")

            data = pd.read_csv(self.data_path)

            self.logger.info("start data pre-process")

            # drop irrelevant tests
            data = data[data.Diagnosis != '?']
            data = data[data.Diagnosis != 'not found']

            width = data.shape[1] - 1

            # over sample positive results
            bool_array = data.iloc[:, -1].values
            row_multiplier = np.fromiter((e == 'True' for e in bool_array), bool)*8 + 1
            data = data.loc[data.index.repeat(row_multiplier)].reset_index(drop=True)

            # split to test and train data sets
            training_set, test_set = train_test_split(data, test_size=0.3, random_state=1)
            x_train = training_set.iloc[:, 2:width].values
            y_train = training_set.iloc[:, -1].values
            x_test = test_set.iloc[:, 2:width].values
            y_test = test_set.iloc[:, -1].values

            # x = df.values  # returns a numpy array

            # data formatting
            x_train = np.nan_to_num(x_train)
            x_test = np.nan_to_num(x_test)

            y_test = np.fromiter((e == 'True' for e in y_test), bool)
            y_train = np.fromiter((e == 'True' for e in y_train), bool)

            # data normalization
            min_max_scaler = preprocessing.MinMaxScaler()
            x_train = min_max_scaler.fit_transform(x_train)
            x_test = min_max_scaler.transform(x_test)

            self.logger.info("start model fitting")

            # initialize SVM and fit training data
            classifier = SVC(kernel='linear', random_state=1)
            classifier.fit(x_train, y_train)

            # Predicting the classes for test set
            y_pred = classifier.predict(x_test)

            # calc confusion matrix

            self.logger.info("calculating resuls")
            cm = confusion_matrix(y_test, y_pred)
            self.logger.info(cm)

            # tp / (tp +fn)
            recall = cm[0, 0] / (cm[0, 1] + cm[0, 0])
            # tp / (tp + fp)
            precision = cm[0, 0] / (cm[1, 0] + cm[0, 0])

            self.logger.info("recall = %s" % recall)
            self.logger.info("precision = %s" % precision)

            # ------------------------- TEST DATA ANALYSIS -------------------#

            # da = DataAnalysis.DataAnalyzer(self.logger, self.database_connector)
            # da.runrun()

            self.logger.info("finish run")
            return True
        except Exception as error:
            self.logger.exception(error)
            return False
