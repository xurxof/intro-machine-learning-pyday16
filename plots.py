
import numpy as np

import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score,confusion_matrix


def show_classifier (clf, X, Y, title):
    from matplotlib.colors import ListedColormap
    from sklearn import neighbors
    
    X= X.to_dense().as_matrix()
    Y= Y.to_dense().as_matrix()
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


    # we create an instance of Neighbours Classifier and fit the data.
    #clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    #clf.fit(X, Y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)

    # plt.show()
    return plt
    


def show_tree (classifier, feat_names, class_names):
    from sklearn.externals.six import StringIO
    from sklearn import tree
    import pydotplus
    from IPython.display import Image  

    dotfile = StringIO()
    tree.export_graphviz(classifier, 
                         out_file=dotfile ,
                         feature_names= feat_names,
                         class_names=class_names,  
                         filled = True
                         )
    pydotplus.graph_from_dot_data(dotfile.getvalue()).write_png("dtree.png")

    return Image(filename='dtree.png') 


def plot_confusion_matrix(y, prediction, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    accuracy = accuracy_score(y, prediction)
    cm = confusion_matrix(y, prediction)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    print ('Accuracy %f' % accuracy)
    print (cm)
    return cm