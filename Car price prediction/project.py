import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle
from sklearn import preprocessing

training_data = np.load('data/training_data.npy')
prices = np.load('data/prices.npy')

# print('The first 4 samples are:\n ', training_data[:4])
# print('The first 4 prices are:\n ', prices[:4])
# # shuffle
training_data, prices = shuffle(training_data, prices, random_state=0)


# 2
def normalize_data(train_data, test_data):
    scaler = preprocessing.StandardScaler()

    if scaler != None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)

        return scaled_train_data, scaled_test_data


# 3
# fold -> 3 samples per fold
num_samples_fold = len(training_data) // 3
print(num_samples_fold)

# impartim train in 3 folds

training_data_1, prices_1 = training_data[:num_samples_fold], \
                            prices[:num_samples_fold]

training_data_2, prices_2 = training_data[num_samples_fold: 2 * num_samples_fold], \
                            prices[num_samples_fold: 2 * num_samples_fold]

training_data_3, prices_3 = training_data[2 * num_samples_fold:], \
                            prices[2 * num_samples_fold:]


def step(train_data, train_labels, test_data, test_labels,model):
    normalized_train, normalized_test = normalize_data(train_data, test_data)
    reg = model.fit(normalized_train, train_labels)
    mae = mean_absolute_error(test_labels, reg.predict(normalized_test))
    mse = mean_squared_error(test_labels, reg.predict(normalized_test))

    return mae, mse

model = LinearRegression()

#Run 1
mae1, mse1 = step(np.concatenate((training_data_1, training_data_3)),
                    np.concatenate((prices_1, prices_3)),
                    training_data_2,
                    prices_2,
                    model)
#Run 2
mae2, mse2 = step(np.concatenate((training_data_1,training_data_2)),
                 np.concatenate((prices_1,prices_2)),
                 training_data_3,
                 prices_3,
                 model)
#Run 3
mae3, mse3 = step(np.concatenate((training_data_2,training_data_3)),
                  np.concatenate((prices_2,prices_3)),
                  training_data_3,
                  prices_3,
                  model)

print("Mae 1 : ", mae1)
print("Mae 2 : ", mae2)
print("Mae 2 : ", mae2)
print("-------------------")
print("Mse 1 : ", mse1)
print("Mse 2 : ", mse2)
print("Mse 3 : ", mse3)

# 3

for alpha_ in [1, 10, 100, 1000]:
    model = Ridge(alpha=alpha_)

    print("alpha = %d -> \n" % alpha_)

    # Run 1
    mae1, mse1 = step(np.concatenate((training_data_1, training_data_3)),
                      np.concatenate((prices_1, prices_3)),
                      training_data_2,
                      prices_2,
                      model)
    # Run 2
    mae2, mse2 = step(np.concatenate((training_data_1, training_data_2)),
                      np.concatenate((prices_1, prices_2)),
                      training_data_3,
                      prices_3,
                      model)
    # Run 3
    mae3, mse3 = step(np.concatenate((training_data_2, training_data_3)),
                      np.concatenate((prices_2, prices_3)),
                      training_data_3,
                      prices_3,
                      model)

    print("Mae 1 : ", mae1)
    print("Mae 2 : ", mae2)
    print("Mae 2 : ", mae2)
    print("-------------------")
    print("Mse 1 : ", mse1)
    print("Mse 2 : ", mse2)
    print("Mse 3 : %f \n" % mse3)

#4
model = Ridge(10)
scaler = preprocessing.StandardScaler()
scaler.fit(training_data)
norm_train = scaler.transform(training_data)
model.fit(norm_train, prices)

print("Coeficientii sunt: ", model.coef_)
print("Bias ul este: ", model.intercept_)

features = ["Year",
            "Kilometers Driven",
            "Mileage",
            "Engine",
            "Power",
            "Seats",
            "Owner Type",
            "Fuel Type",
            "Transmission"]

index_maxim = np.argmax(np.abs(model.coef_))
semnificativ_feature = features[int(index_maxim)]

semnificativ_feature2 = features[(index_maxim + 1)]

index_minim = np.argmin(np.abs(model.coef_))
putinSemnificativ_feature = features[int(index_minim)]

print("Features : ", features)

print("Cel mai semnificativ atribut este: ",semnificativ_feature)
print("Al doilea cel mai semnificativ atribut este: ", semnificativ_feature2)
print("Cel mai putin semnificativ atribut este:", putinSemnificativ_feature)