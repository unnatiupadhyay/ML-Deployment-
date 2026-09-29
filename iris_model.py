from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


# load dataset
iris=load_iris()
x,y=iris.data,iris.target

# train - test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# train model

model=LogisticRegression(max_iter=200)
model.fit(x_train,y_train)

# save model

joblib.dump(model, "iris_model.pkl")
print("model saved as iris_model.pkl")
