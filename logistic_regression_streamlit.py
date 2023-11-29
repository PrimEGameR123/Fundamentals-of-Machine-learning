import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification,make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_initial_graph(dataset, ax):
    """
    Load the initial graph for the given dataset.
    Parameters:
    - dataset (str): The name of the dataset to load. Possible values are 'Binary' and 'Multiclass'.
    - ax (AxesSubplot): The subplot to plot the graph on.
    Returns:
    - x (ndarray): The input data.
    - y (ndarray): The target labels.
    """
    if dataset == 'None':
        pass
    if dataset == 'Binary':
        x, y = make_blobs(n_features=2, centers=2, random_state=6)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap='rainbow')
        return x, y
    elif dataset == 'Multiclass':
        x, y = make_blobs(n_features=2, centers=3, random_state=2)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap='rainbow')
        return x, y

def draw_meshgrid():
    """
    Generate a meshgrid for plotting purposes.

    This function generates a meshgrid using the minimum and maximum values of the
    input array `x`. The meshgrid is created by creating evenly spaced points
    between the minimum and maximum values of each dimension of `x`. The step size
    between each point is set to 0.01.

    Returns:
        aa (ndarray): A 2-dimensional array representing the first dimension of the
            meshgrid.
        bb (ndarray): A 2-dimensional array representing the second dimension of the
            meshgrid.
        input_array (ndarray): A 2-dimensional array representing the meshgrid
            coordinates.

    Example usage:
        aa, bb, input_array = draw_meshgrid()
    """
    a = np.arange(start = x[:, 0].min()-1, stop=x[:, 0].max()+1, step=0.01)
    b = np.arange(start = x[:, 1].min()-1, stop=x[:, 1].max()+1, step=0.01)
    aa, bb = np.meshgrid(a, b)
    input_array = np.array([aa.ravel(), bb.ravel()]).T
    return aa, bb, input_array
plt.style.use('fivethirtyeight')
st.sidebar.markdown("# Logistic Regression Classifier")

dataset = st.sidebar.selectbox("Select dataset type", ('Binary', 'Multiclass'))
penalty = st.sidebar.selectbox("Regularization", ('none','l1', 'l2', 'elasticnet'))
c_input = float(st.sidebar.number_input("C Value", value=1.0))
solver = st.sidebar.selectbox("Solver", ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'))
max_iter = int(st.sidebar.number_input('Max Iterations or epoch', value=100))
multi_class = st.sidebar.selectbox("Multi Class", ('auto', 'ovr', 'multinomial'))
l1_ratio = int(st.sidebar.number_input('L1 Ratio'))

# Load initial graph
fig, ax = plt.subplots()

# plot initial graph
x, y = load_initial_graph(dataset, ax)
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2, random_state=6)
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):
    orig.empty()
    clf = LogisticRegression(penalty=penalty, C=c_input, solver=solver, max_iter=max_iter, multi_class=multi_class, l1_ratio=l1_ratio)
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_ts)
    xx, yy, input_array = draw_meshgrid()
    labels = clf.predict(input_array)
    ax.contourf(xx, yy, labels.reshape(xx.shape), alpha=0.5, cmap = 'rainbow')
    plt.xlabel('col1')
    plt.ylabel('col2')
    orig = st.pyplot(fig)
    st.subheader("Accuracy Score "+str(round(accuracy_score(y_ts, y_pred), 5)))