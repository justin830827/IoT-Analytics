import numpy as np
from hmmlearn.hmm import MultinomialHMM
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chisquare
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

# Task 2 Estimate 𝒑(𝑶|𝝀)


def forward(obs_seq, p, b, pi):
    T = len(obs_seq)
    N = p.shape[0]
    alpha = np.zeros((T, N))
    alpha[0] = pi*b[:, obs_seq[0]-1]
    for t in range(1, T):
        alpha[t] = alpha[t-1].dot(p) * b[:, obs_seq[t]-1]
    return alpha


def viterbi(obs_seq, p, b, pi):
    # returns the most likely state sequence given observed sequence x
    # using the Viterbi algorithm
    T = len(obs_seq)
    N = p.shape[0]
    delta = np.zeros((T, N))
    psi = np.zeros((T, N))
    delta[0] = pi * b[:, obs_seq[0]-1]
    for t in range(1, T):
        for j in range(N):
            delta[t, j] = np.max(delta[t-1] * p[:, j]
                                 ) * b[j, obs_seq[t]-1]
            psi[t, j] = np.argmax(delta[t-1] * p[:, j])
    # backtrack
    states = np.zeros(T, dtype=np.int32)
    states[T-1] = np.argmax(delta[T-1])
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]
    return states + 1


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
    O, Q = generate_observation(1000, p_matrix, b_matrix)

    O_seq = [1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3]
    pi = (1, 0, 0, 0)
    print("\nThe Orginal Observation Sequence O: {}".format(O[:12]))
    print("The probability 𝑝(𝑂|𝜆) is {} with O: {}".format(
        forward(O_seq, p_matrix, b_matrix, pi)[-1].sum(), O_seq))

    print("\nThe Orginal Sequence Q: {}".format(Q[:12]))
    print("The Most Probable Sequence Q: {} with O: {}".format(
        list(viterbi(O_seq, p_matrix, b_matrix, pi)), O_seq))

    obersvations = LabelEncoder().fit_transform(O)
    model = MultinomialHMM(n_components=4)
    model.fit(np.atleast_2d(obersvations))
    est_pi = model.startprob_
    est_p = model.transmat_
    est_b = model.emissionprob_
    print("\nThe estimated transition matrix P:\n {}".format(est_p))
    print("\nThe estimated event matrix B:\n {}".format(est_b))
    print("\nThe estimated start probability pi:\n {}".format(est_pi))

    _, p = chisquare(p_matrix, est_p, axis=None)
    print("\np-value of transition matrix P: {}".format(p))

    _, p = chisquare(b_matrix, est_b, axis=None)
    print("p-value of event matrix B: {}".format(p))

    _, p = chisquare(pi, est_pi, axis=None)
    print("p-value of start probability pi: {}".format(p))


if __name__ == "__main__":
    main()
