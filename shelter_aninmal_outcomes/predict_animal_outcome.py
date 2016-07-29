__author__ = 'Rays'

import cPickle
import csv
import datetime
import itertools
import numpy as np
import pandas as pd
import sklearn.cross_validation as cv
import sklearn.externals.joblib as joblib
import sklearn.grid_search as grid_search
import sklearn.metrics as metrics
import sklearn.svm as svm
import timeit
import sklearn.ensemble as ensemble


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

WEEKS_IN_A_MONTH = 4.35
with open("breed_list.pkl", "r") as breed_list_file:
    BREED_LIST = cPickle.load(breed_list_file)
with open("color_list.pkl", "r") as color_list_file:
    COLOR_LIST = cPickle.load(color_list_file)

OUTCOME_TYPE = ["Adoption",	"Died", "Euthanasia", "Return_to_owner", "Transfer"]

def map_gender(s):
    if "Male" in s:
        return 1
    elif "Female" in s:
        return -1
    else:
        return 0


def map_birthability(s):
    if "Intact" in s:
        return 1
    elif "Spayed" in s or "Neutered" in s:
        return -1
    else:
        return 0


def map_age_in_month(s):
    terms = s.split()
    if "week" in terms[1]:
        ratio = 1/WEEKS_IN_A_MONTH
    elif "year" in terms[1]:
        ratio = 12
    else:
        ratio = 1
    return int(terms[0]) * ratio


def shape_list(data):
    return np.asarray(list(itertools.chain(*data))).reshape(data.shape[0], len(data[0]))


def extract_breeds(s):
    if "Mix" in s:
        breed1 = s[0:(s.index("Mix") - 1)]
        breed2 = "Mix"
    elif "/" in s and s != "Black/Tan Hound":
        try:
            index = s.index("Black/Tan Hound")
            if index == 0:
                breed1 = "Black/Tan Hound"
                breed2 = s[len("Black/Tan Hound")+1:]
            else:
                breed1 = s[0:(s.index("Black/Tan Hound") - 1)]
                breed2 = "Black/Tan Hound"
        except ValueError:
            breed1, breed2 = s.split("/")
            if breed2 == breed1:
                breed2 = "None"
    else:
        breed1 = s
        breed2 = "None"
    return breed1, breed2


def map_pure_breed(s):
    breed1, breed2 = extract_breeds(s)
    return 1 if breed2 == "None" else 0


def map_breed_1(s):
    breed1, breed2 = extract_breeds(s)
    return breed1


def map_breed_2(s):
    breed1, breed2 = extract_breeds(s)
    return breed2


def map_breed_mix(s):
    breed1, breed2 = extract_breeds(s)
    breed_feature = np.asarray([1 if (b == breed1 or b == breed2) else 0 for b in BREED_LIST])
    #assert(sum(breed_feature) == 2)
    return breed_feature


def extract_color(s):
    if "/" in s:
        color1, color2 = s.split("/")
        if color1 == color2:
            color2 = "None"
    else:
        color1 = s
        color2 = "None"
    return color1, color2


def map_pure_color(s):
    color1, color2 = extract_color(s)
    return 1 if color2 == "None" else 0


def map_color_mix(s):
    color1, color2 = extract_color(s)
    color_feature = np.asarray([1 if (b == color1 or b == color2) else 0 for b in COLOR_LIST])
    #assert(sum(color_feature) == 2)
    return color_feature


def map_color1(s):
    color1, color2 = extract_color(s)
    return color1


def map_color2(s):
    color1, color2 = extract_color(s)
    return color2


def load_training_data():
    return load_data("data/train.csv")


def load_testing_data():
    return load_data("data/test.csv")


def load_data(fn):
    df = pd.read_csv(fn)
    df["SexuponOutcome"] = df["SexuponOutcome"].fillna("Unknown")
    # all unknown age seems be around 2/9-2/21/2016.
    # it seems the AgeuponOutcome == NaN is noise and most in training ended up being transferred,
    # hence we should bluntly predict transfer for the records in test.csv that don't have age.
    df = df[df["AgeuponOutcome"].notnull()]

    # convert to new features
    df["Gender"] = df["SexuponOutcome"].map(map_gender)
    df["Birthability"] = df["SexuponOutcome"].map(map_birthability)
    df["AgeInMonth"] = df["AgeuponOutcome"].map(map_age_in_month)
    # 8-16 weeks are the best, but since the accuracy of age is by months once after 8 weeks, we can use 4 months = 16 weeks
    df["BestAdoptionAge"] = df["AgeInMonth"].map(lambda x: 1 if (8 / WEEKS_IN_A_MONTH * 0.95) <= x <= 4 else 0)
    df["Purebreed"] = df["Breed"].map(map_pure_breed)
    df["Breed1"] = df["Breed"].map(map_breed_1)
    df["Breed2"] = df["Breed"].map(map_breed_2)
    df["HasName"] = df["Name"].map(lambda x: 1 if pd.notnull(x) else 0)
    df["DateTime"] = df["DateTime"].map(lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))
    df["SingeColor"] = df["Color"].map(map_pure_color)
    df["Color1"] = df["Color"].map(map_color1)
    df["Color2"] = df["Color"].map(map_color2)
    #Mon = 0, Sun = 6
    df["DayOfWeek"] = df["DateTime"].map(lambda d: d.weekday())
    # df["InMorning"] = df["DateTime"].map(lambda d: 1 if d.hour < 13 else 0)
    return df


def extract_features(df, extract_y=True):
    x = df[["Gender", "Birthability", "AgeInMonth", "BestAdoptionAge", "DayOfWeek",
            "Purebreed", "HasName", "SingeColor"]].values
    x = np.append(x,
                  shape_list(df["Breed"].map(map_breed_mix).values),
                  axis=1)
    x = np.append(x,
                  shape_list(df["Color"].map(map_color_mix).values),
                  axis=1)
    if extract_y:
        y = df["OutcomeType"].map(lambda s: np.asarray(OUTCOME_TYPE.index(s))).values
    else:
        y = []
    return x, y


def svm_train(df):
    x, y = extract_features(df)
    x_train, x_test, y_train, y_test = cv.train_test_split(x, y, test_size=0.3, random_state=42)
    tuned_parameters = [{'kernel': ['rbf'], 'C': [10, 100, 300], 'gamma': [0.003]}] #C 10,100,300
    scores = ['accuracy']
    for score in scores:
        t0 = timeit.default_timer()
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = grid_search.GridSearchCV(svm.SVC(C=1, decision_function_shape='ovo'), tuned_parameters, cv=5,
                                       scoring='%s' % score)
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)
        print(metrics.classification_report(y_true, y_pred))
        print("Total runtime: %.4f" % (timeit.default_timer() - t0))
        print()


def rf_train(df):
    x, y = extract_features(df)
    x_train, x_test, y_train, y_test = cv.train_test_split(x, y, test_size=0.3, random_state=42)
    tuned_parameters = [{'n_estimators': [10, 50, 100], 'min_samples_leaf': [5, 10], 'min_samples_split': [5, 10, 20]}] #C 10,100,300
    scores = ['accuracy']
    for score in scores:
        t0 = timeit.default_timer()
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = grid_search.GridSearchCV(ensemble.RandomForestClassifier(n_jobs=3, min_samples_leaf=1, min_samples_split=2), tuned_parameters, cv=5,
                                       scoring='%s' % score)
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)
        print(metrics.classification_report(y_true, y_pred))
        print("Total runtime: %.4f" % (timeit.default_timer() - t0))
        print()


def ada_train(df):
    x, y = extract_features(df)
    x_train, x_test, y_train, y_test = cv.train_test_split(x, y, test_size=0.3, random_state=42)
    tuned_parameters = [{'n_estimators': [10, 50, 100, 150], 'learning_rate': [0.1, 1, 3, 10]}]
    scores = ['accuracy']
    for score in scores:
        t0 = timeit.default_timer()
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = grid_search.GridSearchCV(ensemble.AdaBoostClassifier(random_state=42), tuned_parameters, cv=5,
                                       scoring='%s' % score)
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)
        print(metrics.classification_report(y_true, y_pred))
        print("Total runtime: %.4f" % (timeit.default_timer() - t0))
        print()


def nn_train(df):
    x, y = extract_features(df)
    x_train, x_cv, y_train, y_cv = cv.train_test_split(x, y, test_size=0.3, random_state=42)
    batch_size = 1000
    nb_epoch = 50
    hidden_unit_width = 500
    drop_out_rate = 0.25
    model = Sequential()
    model.add(Dense(input_dim=x_train.shape[1], output_dim=hidden_unit_width))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out_rate))
    model.add(Dense(input_dim=hidden_unit_width, output_dim=hidden_unit_width))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out_rate))
    model.add(Dense(input_dim=hidden_unit_width, output_dim=hidden_unit_width))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out_rate))
    model.add(Dense(output_dim=5))
    model.add(Activation('softmax'))
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(x_cv, y_cv),
              shuffle=True)


def train_and_save_model(df_train, fn):
    x_train, y_train = extract_features(df_train)
    clf = svm.SVC(C=100, gamma=0.003, decision_function_shape="ovo")
    clf.fit(x_train, y_train)
    joblib.dump(clf, fn)


def predict(df_test, fn):
    clf = joblib.load(fn)
    x_test, y_test = extract_features(df_test, extract_y=False)
    df_test["prediction"] = clf.predict(x_test)
    return df_test


def save_results(df, fn):
    with open(fn, "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["ID", "Adoption",	"Died", "Euthanasia", "Return_to_owner", "Transfer"])
        id_name = "ID" if "ID" in df else "AnimalID"
        for a_id, prediction in df[[id_name, "prediction"]].values:
            row = [a_id]
            for index in range(len(OUTCOME_TYPE)):
                row.append(1 if prediction == index else 0)
            writer.writerow(row)
