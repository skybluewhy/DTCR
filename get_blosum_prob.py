from sklearn.preprocessing import normalize
import numpy as np
import torch
# from sequence_models.constants import MASK, MSA_PAD, MSA_ALPHABET, MSA_AAS, GAP, START, STOP, SEP, AAINDEX_ALPHABET, AMB_AAS, OTHER_AAS



def loadMatrix(path):
    """
    Taken from https://pypi.org/project/blosum/
    Edited slightly from original implementation

    Reads a Blosum matrix from file. Changed slightly to read in larger blosum matrix
    File in a format like:
        https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62
    Input:
        path: str, path to a file.
    Returns:
        blosumDict: Dictionary, The blosum dict
    """

    with open(path, "r") as f:
        content = f.readlines()

    blosumDict = {}

    header = True
    for line in content:
        line = line.strip()

        # Skip comments starting with #
        if line.startswith(";"):
            continue

        linelist = line.split()

        # Extract labels only once
        if header:
            labelslist = linelist[:20] + linelist[-2:]
            header = False

            # Check if all AA are covered
            #if not len(labelslist) == 25:
            #    print("Blosum matrix may not cover all amino-acids")
            continue

        linelist = [linelist[0]] + linelist[1:21] + linelist[-2:]
        if not len(linelist) == len(labelslist) + 1:
            print(len(linelist), len(labelslist))
            # Check if line has as may entries as labels
            raise EOFError("Blosum file is missing values.")

        # Add Line/Label combination to dict
        if linelist[0] not in labelslist:
            continue
        for index, lab in enumerate(labelslist, start=1):
            blosumDict[f"{linelist[0]}{lab}"] = float(linelist[index])

    # Check quadratic
    if not len(blosumDict) == len(labelslist) ** 2:
        print(len(blosumDict), len(labelslist))
        raise EOFError("Blosum file is not quadratic.", len(blosumDict), len(labelslist)**2)
    return blosumDict


def softmax(x):
    """
    Compute softmax over x
    """
    return np.exp(x)/np.sum(np.exp(x),axis=0)


def double_stochastic(q):
    q_norm = normalize(q, axis=1, norm='l1')
    while not np.isclose(np.min(np.sum(q_norm, axis=0)), 1): # only checking that one value converges to 1 (prob best to do all 4 min/max)
        q_norm = normalize(q_norm, axis=0, norm='l1')
        q_norm = normalize(q_norm, axis=1, norm='l1')
    return q_norm


# BLOSUM_AAS = AAINDEX_ALPHABET + AMB_AAS
BLOSUM_ALPHABET = "ARNDCQEGHILKMFPSTWYV#-"
# ALPHABET = "-ARNDCQEGHILKMFPSTWYV#"
ALPHABET = "-CWVAHTEKNPILSDGQRYFM#"
# {"PAD": 0, "C": 1, "W": 2, "V": 3, "A": 4, "H": 5, "T": 6, "E": 7, "K": 8, "N": 9, "P": 10, "I": 11, "L": 12, "S": 13, "D": 14, "G": 15, "Q": 16, "R": 17, "Y": 18, "F": 19, "M": 20, "MASK": 21}
def q_blosum():
    all_aas = list(BLOSUM_ALPHABET)
    matrix = loadMatrix("./data/blosum62.mat")
    matrix_dict = dict(matrix)

    # alphabet = list("".join(MSA_ALPHABET))
    alphabet = ALPHABET
    a_to_i = {u: i for i, u in enumerate(alphabet)}

    q = np.array([i for i in matrix_dict.values()])
    q = q.reshape((len(all_aas), len(all_aas)))
    q = softmax(q)
    q = double_stochastic(q)
    q = torch.tensor(q)
    # REORDER BLOSUM MATRIX BASED ON MSA_ALPHABET (self.alphabet, self.a_to_i)
    new_q = q.clone()
    i2_to_a = np.array(list(BLOSUM_ALPHABET))
    for i, row in enumerate(new_q):
        for j, value in enumerate(row):
            ind1, ind2 = [i, j]
            key = i2_to_a[ind1], i2_to_a[ind2]
            new1, new2 = [a_to_i[k] for k in key]
            new_q[new1, new2] = q[ind1, ind2]
    return new_q


new_q = q_blosum()
# print(new_q)


def _beta_schedule(num_timesteps, schedule='linear', start=1e-5, end=0.999, max=8):
    """
    Variance schedule for adding noise
    Start/End will control the magnitude of sigmoidal and cosine schedules.
    """
    if schedule == 'linear':
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == 'sohl-dickstein':
        betas = torch.linspace(0,num_timesteps-1, num_timesteps)
        betas = 1/(num_timesteps - betas + 1)
    elif schedule == "cosine":
        betas = torch.linspace(np.pi / 2, 0, num_timesteps)
        betas = torch.cos(betas) * (end - start) + start
    elif schedule == "exp":
        betas = torch.linspace(0, max, num_timesteps)
        betas = torch.exp(betas) * (end - start) + start
    else:
        print("Must select a valid schedule; ['linear', 'sohl-dickstein', 'cosine', 'exp']")
    return betas

def cumprod_matrix(a):
    """
    Takes a list of transition matrices and ouputs a list of the cumulative products (Q_bar) at each timestep
    """
    a_bar = [a[0]]  # initialize w/ first item in list
    start = a[0]
    for i in range(len(a) - 1):
        a_prod_temp = torch.mm(start, a[i + 1])
        start = a_prod_temp
        a_bar.append(a_prod_temp)  # update start
    return a_bar

def q_random_schedule(timesteps=500, schedule='sohl-dickstein'):
    print(schedule)
    all_aas = list(BLOSUM_ALPHABET)
    K = len(all_aas)
    betas = _beta_schedule(timesteps, schedule=schedule)
    Q_t = []  # scheduled matrix
    for i in range(len(betas)):
        q_non_diag = torch.ones((K, K)) / K * betas[i]
        norm_constant = (1 - (q_non_diag).sum(axis=0))
        q_diag = torch.tensor(np.identity(K)) * norm_constant
        R = q_diag + q_non_diag
        Q_t.append(R)
    Q_prod = cumprod_matrix(Q_t)
    Q_prod = torch.stack(Q_prod)  # cumprod of matrices
    Q_t = torch.stack(Q_t)  # scheduled matrix
    return Q_prod, Q_t

def q_blosum_schedule(timesteps=500, schedule='exp', max=6):
    """
    betas = 'exp' use exp scheme for beta schedule
    """
    print(schedule)
    all_aas = list(BLOSUM_ALPHABET)
    K = len(all_aas)
    q = q_blosum()
    betas = _beta_schedule(timesteps, schedule=schedule, max=max)
    betas = betas / betas.max() + 1/timesteps
    Q_t = []  # scheduled matrix
    for i in range(timesteps):
        q_non_diag = torch.ones((K,K)) * q * betas[i]
        norm_constant = (1 - (q_non_diag).sum(axis=0))
        q_diag = torch.tensor(np.identity(K)) * norm_constant
        R = q_diag + q_non_diag
        Q_t.append(R)
    Q_prod = cumprod_matrix(Q_t)
    Q_prod = torch.stack(Q_prod) # cumprod of matrices
    Q_t = torch.stack(Q_t) # scheduled matrix
    return Q_prod, Q_t


# q_random_schedule()
# q_blosum_schedule()
