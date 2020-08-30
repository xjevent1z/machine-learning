"""A package"""
import numpy as np
import pandas as pd


class Income:
    """Predict the probability of income > 50k"""
    x_train = None
    y_train = None
    y_hat = None

    def encode(self, path):
        """read data"""
        df = pd.read_csv(path)
        discrete_cols = [
            col for col in df.columns if df[col].dtype == 'object' and col != 'income'
        ]
        continuous_cols = [
            col for col in df.columns if df[col].dtype != 'object'
        ]
        y_cols = [col for col in df.columns if col == 'income']
        x_train = pd.get_dummies(
            df[discrete_cols]).join(df[continuous_cols])
        y_train = pd.get_dummies(df[y_cols])

        x_train.to_csv('classification/myXTrain.csv', index=False)
        y_train.to_csv('classification/myYTrain.csv', index=False)

        dfX = pd.read_csv('classification/myXTrain.csv')
        dfY = pd.read_csv('classification/myYTrain.csv')
        self.x_train = np.array(dfX.values, dtype=float)
        self.y_train = np.array(dfY.values, dtype=float)

    def training(self):
        self.y_hat = self.weight * self.x_train + self.bias
        print(self.y_hat)

    @property
    def cov_1(self):
        tmp = np.zeros((self.x_train.shape[1], self.x_train.shape[1]))
        cov = list()
        for col, mu in zip(self.x_train.T, self.u_1):
            cov.append(np.sum(np.dot((col - mu), (col - mu).T)) /
                       self.x_train.shape[0])
        tmp += np.array(cov, dtype=float)
        return tmp

    @property
    def cov_2(self):
        tmp = np.zeros((self.x_train.shape[1], self.x_train.shape[1]))
        cov = list()
        for col, mu in zip(self.x_train.T, self.u_2):
            cov.append(np.sum(np.dot((col - mu), (col - mu).T)) /
                       self.x_train.shape[0])
        tmp += np.array(cov, dtype=float)
        return tmp

    @property
    def cov(self):
        return self.cov_1 / \
            np.count_nonzero(
                self.y_train[:, 0]) + self.cov_2 / np.count_nonzero(self.y_train[:, 1])

    @property
    def u_1(self):
        mu = list()
        for col in self.x_train.T:
            _class = col * self.y_train[:, 0]
            _class[_class == 0] = np.nan
            mu.append(np.nanmean(_class, axis=0, dtype=float))
        return np.array(mu, dtype=float)

    @property
    def u_2(self):
        mu = list()
        for col in self.x_train.T:
            _class = col * self.y_train[:, 1]
            _class[_class == 0] = np.nan
            mu.append(np.nanmean(_class, axis=0, dtype=float))
        return np.array(mu, dtype=float)

    @property
    def weight(self):
        return np.dot((self.u_1 - self.u_2).T, np.linalg.inv(self.cov))

    @property
    def bias(self):
        return (-1/2) * np.dot(np.dot(self.u_1.T, np.linalg.inv(self.cov)), self.u_1) + \
            (1/2) * np.dot(np.dot(self.u_2.T, np.linalg.inv(self.cov)), self.u_2) + \
            np.log2(np.count_nonzero(self.y_train[:, 0])) - \
            np.log2(np.count_nonzero(self.y_train[:, 1]))
