import numpy as np
from hmmlearn.hmm import MultinomialHMM
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import math
np.random.seed(200263453)


def normalization(matrix):
    print("Checking each row's sum of prob. is equal to 1.")
    for i in range(len(matrix)):
        total = sum(matrix[i])
        for j in range(len(matrix[i])):
            matrix[i][j] = matrix[i][j] / total
        print("Sum of row[{}] is {}".format(i, sum(matrix[i])))
    return matrix

# Identify observation


def get_observation(state, b):
    v = [1, 2, 3]  # states
    return np.random.choice(v, 1, p=b[state-1])[0]

# Get the shift state


def get_shift_state(state, p):
    S = [1, 2, 3, 4]
    return np.random.choice(S, 1, p=p[state-1])[0]


def generate_observation(n, p, b):
    Q = [1]
    O = []
    O.append(get_observation(Q[-1], b))
    while len(O) < n:
        Q.append(get_shift_state(Q[-1], p))
        O.append(get_observation(Q[-1], b))
    return O, Q


def AIC(logL, p):
    return -2 * logL + 2 * p


def BIC(logL, observations, p):
    T = len(observations)
    return -2 * logL + p * math.log(T)


def compute_p(n, m):
    return m ** 2 + n * m - 1


def plot(y, label):
    x = np.arange(2, 30)
    plt.plot(x, y, marker='o', color='blue', linewidth=2, label=label)
    plt.xlabel('Number of states')
    plt.ylabel(label)
    plt.title('Selection of number of states -{}'.format(label))
    plt.savefig('./results/{}.png'.format(label))
    plt.show()


def main():
    rand_p_matrix = np.random.rand(4, 4)
    rand_b_matrix = np.random.rand(4, 3)

    print("\nGernerating p matrix...............")
    p_matrix = normalization(rand_p_matrix)
    print(p_matrix)

    print("\nGernerating b matrix...............")
    b_matrix = normalization(rand_b_matrix)
    print(b_matrix)

    # Generate 1000 observations
    O, _ = generate_observation(1000, p_matrix, b_matrix)

    # training the selection of number of states
    aic = []
    bic = []
    likelihood = []
    m = 3
    print("\nTraining the HMM for selection of number of states........")
    for n in range(2, 30):
        observations = LabelEncoder().fit_transform(O)
        model = MultinomialHMM(n_components=n, random_state=200263453)
        model.fit(np.atleast_2d(observations))
        logL = model.score(np.atleast_2d(observations))
        p = compute_p(n, m)
        a = AIC(logL, p)
        b = BIC(logL, observations, p)
        likelihood.append(logL)
        aic.append(a)
        bic.append(b)
    plot(aic, 'AIC')
    plot(bic, 'BIC')
    plot(likelihood, 'Log likelihood')


if __name__ == "__main__":
    main()
