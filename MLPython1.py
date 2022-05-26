from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()

#print(iris.DESCR)

features=iris.data
labels=iris.target

Classifier=KNeighborsClassifier()
Classifier.fit(features,labels)

a=int(input("Enter the Sepal Length in cm :  "))
b=int(input("Enter the Sepal Width in cm :  "))
c=int(input("Enter the Petal Length in cm :  "))
d=int(input("Enter the Petal Width in cm :  "))

predict=Classifier.predict([[a,b,c,d]])

if predict==[0]:
    print("The Iris class is Setosa")
elif predict==[1]:
    print("The Iris class is Versicolour")
else:
    print("The Iris class is Virginica")
    
    