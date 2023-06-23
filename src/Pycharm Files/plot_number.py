#Visualization
import matplotlib.pyplot as plt

def plot_digit_funct(i, X, y):
    # Testing label-image
    # print("The label number is: ", y_train[i])



    plt.title(y[i])
    plt.imshow(X[i], cmap='binary')
    plt.show()

