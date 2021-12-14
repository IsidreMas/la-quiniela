import pickle
import time
import datetime
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class QuinielaModel:

    def train(self, train_data):
        def get_year(date, season):
            date = date.split("/")
            if(season.split("-")[0][-2:] == date[2]):
                return int(season.split("-")[0])
            elif(season.split("-")[1][-2:] == date[2]):
                return int(season.split("-")[1])
        def goal_diff(score):
            score = score.split(":")
            return int(score[0])-int(score[1])

        def result(score):
            score = score.split(":")
            if score[0]>score[1]:
                return '1'
            elif score[1]>score[0]:
                return '2'
            else:
                return 'X'
        # Preprocessing

        df = train_data.drop('time', 1).dropna(subset=['score'])
        df['season_start'] = df.apply(lambda row: int(row.season.split("-")[0]), axis=1)
        df['day'] = df.apply(lambda row: int(row.date.split("/")[1]), axis=1)
        df['month'] = df.apply(lambda row: int(row.date.split("/")[0]), axis=1)
        df['year'] = df.apply(lambda row: get_year(row.date, row.season), axis=1)
        df['timestamp'] = df.apply(lambda row: time.mktime(datetime.datetime(year=row.year, month=row.month, day=row.day).timetuple()), axis=1)
        df['result'] = df.apply(lambda row: result(row.score), axis=1)
        df = df.drop(['score','season','date'],1)
        df = df.sort_values(by='timestamp')
        model_df = df.join( pd.get_dummies(df[['home_team', 'away_team']])).drop(['home_team', 'away_team'], 1)
        print(model_df)

        X_train = model_df.drop('result',1)
        y_train = model_df['result']

        scaler = preprocessing.MinMaxScaler()
        print(scaler.fit(X_train))
        X_train_norm = scaler.transform(X_train)

        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(max_depth=50, random_state=0)
        self.clf.fit(X_train_norm, y_train)
        pass

    def predict(self, predict_data):

        def get_year(date, season):
            date = date.split("/")
            if(season.split("-")[0][-2:] == date[2]):
                return int(season.split("-")[0])
            elif(season.split("-")[1][-2:] == date[2]):
                return int(season.split("-")[1])
        def goal_diff(score):
            score = score.split(":")
            return int(score[0])-int(score[1])

        def result(score):
            score = score.split(":")
            if score[0]>score[1]:
                return '1'
            elif score[1]>score[0]:
                return '2'
            else:
                return 'X'

        df = predict_data.drop('time', 1).dropna(subset=['score'])
        df['season_start'] = df.apply(lambda row: int(row.season.split("-")[0]), axis=1)
        df['day'] = df.apply(lambda row: int(row.date.split("/")[1]), axis=1)
        df['month'] = df.apply(lambda row: int(row.date.split("/")[0]), axis=1)
        df['year'] = df.apply(lambda row: get_year(row.date, row.season), axis=1)
        df['timestamp'] = df.apply(lambda row: time.mktime(datetime.datetime(year=row.year, month=row.month, day=row.day).timetuple()), axis=1)
        df = df.drop(['score','season','date'],1)
        df = df.sort_values(by='timestamp')
        predict_df = df.join( pd.get_dummies(df[['home_team', 'away_team']])).drop(['home_team', 'away_team'], 1)

        scaler = preprocessing.MinMaxScaler()
        print(predict_df)
        print(scaler.fit(predict_df))
        X_predict_norm = scaler.transform(predict_df)
        # Do something here to predict
        print(self.clf.predict(X_predict_norm))

        return ["X" for _ in range(len(predict_data))]

    @classmethod
    def load(cls, filename):
        """ Load model from file """
        with open(filename, "rb") as f:
            model = pickle.load(f)
            assert type(model) == cls
        return model

    def save(self, filename):
        """ Save a model in a file """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
