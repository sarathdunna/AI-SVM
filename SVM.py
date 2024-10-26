import numpy as np
from cvxopt import solvers
import cvxopt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SVM:
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)
    
    def __init__(self, kernel_str='linear', C=1.0):
        if kernel_str == 'linear':
            self.kernel = self.linear_kernel
        else:
            raise ValueError('Only linear kernel is supported')
        self.C = float(C)
        self.kernel_str = kernel_str
        
    def fit(self, X, y):
        num_samples, num_features = X.shape
        
        # Compute kernel matrix
        kernel_matrix = self.kernel(X, X)
        
        # Set up the quadratic programming problem
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix)
        q = cvxopt.matrix(-np.ones(num_samples))
        A = cvxopt.matrix(y.reshape(1, -1))
        b = cvxopt.matrix(0.0)
        
        # Inequality constraints: 0 ≤ α_i ≤ C
        G = cvxopt.matrix(np.vstack((-np.eye(num_samples), np.eye(num_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(num_samples), np.ones(num_samples) * self.C)))
        
        # Solve QP problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        
        # Extract Lagrange multipliers
        alphas = np.array(solution['x']).flatten()
        
        # Find support vectors
        sv_threshold = 1e-5
        sv_indices = alphas > sv_threshold
        self.alphas = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        
        # Compute weights for linear kernel
        self.w = np.zeros(num_features)
        for n in range(len(self.alphas)):
            self.w += self.alphas[n] * self.support_vector_labels[n] * self.support_vectors[n]
        
        # Compute bias
        margin_vectors = (alphas > sv_threshold) & (alphas < self.C - sv_threshold)
        if np.any(margin_vectors):
            self.b = np.mean(y[margin_vectors] - np.dot(X[margin_vectors], self.w))
        else:
            # If no margin vectors, use average over support vectors
            self.b = np.mean(y[sv_indices] - np.dot(self.support_vectors, self.w))
            
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

def preprocess_data(filename='pulsar_star_dataset.csv'):
    # Load and preprocess data
    df = pd.read_csv(filename)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    y[y == 0] = -1  # Convert to -1/1 labels
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize using training statistics
    mean_train = X_train.mean(axis=0)
    std_train = X_train.std(axis=0)
    X_train = (X_train - mean_train) / std_train
    X_test = (X_test - mean_train) / std_train
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Load and preprocess data
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # For faster testing, use smaller subset
    X_train, y_train = X_train[:800], y_train[:800]
    X_test, y_test = X_test[:200], y_test[:200]
    
    # Test different C values
    C_values = [0.1, 1, 10, 100, 1000]
    accuracies = {}
    
    for C in C_values:
        svm = SVM('linear', C=C)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracies[C] = accuracy_score(y_test, y_pred)
    
    print("Accuracies for different C values:")
    for C, acc in accuracies.items():
        print(f"C = {C:<6} Accuracy = {acc:.3f}")