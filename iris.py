import csv
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
file = open("Iris.csv")
csv_data = csv.reader(file)
data = []
for i in csv_data:
    data.append(i)
file.close()
data_iris = []
for i in range(1, 151):
    data_iris.append(data[i])

# Inputs
x = []
# Outputs
y = []
for i in data_iris:
    x.append([float(i[1]), float(i[2]), float(i[3]), float(i[4])])
    if i[5] == 'Iris-setosa':
        y.append([1, 0, 0])
    elif i[5] == 'Iris-versicolor':
        y.append([0, 1, 0])
    else:
        y.append([0, 0, 1])

model = Sequential()
model.add(Dense(3, input_dim=4, activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(3, activation='sigmoid'))
learning_rate = 0.005
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()

x = np.array(x)
y = np.array(y)

model.fit(x, y, epochs=300, batch_size=15)

prediction_1 = model.predict(x[1:2])
print(f"{prediction_1[0][0]*100:.2f}% Iris-setosa")

prediction_51 = model.predict(x[51:52])
print(f"{prediction_51[0][1]*100:.2f}% Iris-versicolor")

prediction_101 = model.predict(x[101:102])
print(f"{prediction_101[0][2]*100:.2f}% Iris-virginica")