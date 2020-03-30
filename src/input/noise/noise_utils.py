"""
noise_utils.py
====================================
Module containing utilities related to noise.
"""
import numpy as np
import gssl.graph.gssl_utils as gutils
from _heapq import heappush
from builtins import isinstance

def uniform_noise_transition_prob_mat(Y,p):
    """ Obtains the transition probabilities between each class for uniform label noise.
    
    Args:
        Y (`[NDArray[int].shape[N,C]`) : Matrix encoding initial beliefs.
        p (float): The percentage of labels to be flipped
    
    Returns:
        `[NDArray[int].shape[C,C]`: the transition probability matrix.
    
    Raises:
        ValueError: if `p` is an invalid percentage.
    
    """
    c = Y.shape[1]
    if 0 > p or p > 1:
        raise ValueError("Invalid percentage")
    A = np.zeros(shape=(c,c))
    for i in range(c):
        for j in range(c):
            A[i,j] = (1-p) if i == j else p/(c-1)
    return A


def transition_count_mat(Y,A):
    """ Obtains a transition count matrix for uniform noise,
    indicating how many instances should have their label flipped and to which class.
    
    Specifically, this returns a matrix M such that :math:`M[i,j]` is the number of instances of i-th class to be swapped
    to the j-th class.
    
    Args:
        Y (`[NDArray[int].shape[N,C]`) : Matrix encoding initial beliefs.
        A (`[NDArray[int].shape[C,C]`): Transition probabilities between each class.
    Returns:
        `[NDArray[int].shape[C,C]`: the transition count matrix.
    
    Raises:
        ValueError: if `p` is an invalid percentage.
    
    """
    c = Y.shape[1]
    class_freq = [int(round(sum(Y[:,i]))) for i in range(c)]
    num_clean = int(np.round(sum([class_freq[i]*A[i,i] for i in range(c)])))
    
    print("NUM CLEAN:{};NUM NOISY:{};TOTAL:{}".format(num_clean,np.sum(class_freq)-num_clean,np.sum(class_freq) ))
    
    
    
    """ Little procedure that allocates clean labels according to prob. of each diagonal entry """
    import heapq
    B = np.zeros((c,)) #Soon to be our diagonal
    H = [(-A[i,i] * class_freq[i],i) for i in range(c)]
    heapq.heapify(H)
    
    for i in range(num_clean):
        x = heapq.heappop(H)
        id = x[1]
        val = x[0]
        B[id] += 1
        heapq.heappush(H,(val+1,id))
        
    
    """ We approximate the "most commonly observed" scenario by rounding """
    #Fix possible isues
    for i in range(c):
        A[i,:] = np.round(A[i,:] * (class_freq[i] - num_clean))
        A[i,i] = B[i]
    
    
    A = A.astype(np.int32)
    
    observed_class_counts = np.asarray([A[i,i] for i in range(c)])
    expected_class_counts = np.asarray([np.sum(np.argmax(Y,axis=1) == i) for i in range(c)])
    
    
    INFTY = A.shape[0]**2
    for i in range(c):
        S = sum(A[i,:]) - class_freq[i]
        if S > 0:
            for s in range(S):
                diff = observed_class_counts-expected_class_counts
                diff[i] = -INFTY
                diff[A[i,:]==0] = -2*INFTY
                j = np.random.choice(np.flatnonzero(diff == diff.max()))
                observed_class_counts[j] -= 1
                A[i,j] -= 1
        elif S < 0: 
            for s in range(-S):
                diff = expected_class_counts-observed_class_counts
                diff[i] = -INFTY
                j = np.random.choice(np.flatnonzero(diff == diff.max()))
                observed_class_counts[j] += 1
                A[i,j] += 1
        assert sum(A[i,:]) - class_freq[i] == 0
    
    return A


"""
    Picks randomly a number of 
    
    Args:
        P (`[NDArray[float].shape[N]`) : Contains probability of picking each index
        M (`[NDArray[float].shape[N]`) : Contains maximum amount that can be picked from index
        t (int) : number of times to pick indexes.
        
    Returns:
        `List[int].shape[t]` : list of picked indexes 
        
    
"""
def __pick_from(P,t,M = None):
    
    if t == 0:
        return []
    
    def normalize_p(P):
        P = P.astype(np.float32)
        P = P /np.sum(P)
        P = np.round(P,5)
        P[np.argmax(P)] += (1.0 - np.sum(P) )
        return P
        eps =np.finfo(np.float32).eps        
        while not np.sum(P) == 1.0:
            ch = np.random.choice(P.shape[0])
            sgn = np.sign(1.0-np.sum(P))
            if sgn > 0:
                P[ch] += eps
            else:
                P[ch] -= eps
            P[ch] = min(P[ch],1.0)
            P[ch] = max(P[ch],0.0)
        return P
    
    if isinstance(P,list):
        P = np.asarray(P)
    if isinstance(M,list):
        M = np.asarray(M)
        
    
    if M is None:
        M = np.inf * np.ones((P.shape[0],))
    
    
    
    if np.sum(P) == 0:
        raise ValueError("Cannot have 0 probability for every pick")
    
    if P.shape[0] != M.shape[0] or not P.ndim == 1 or not  M.ndim == 1:
        raise ValueError("Wrong shape for probability or maximum")
    
    
    P[np.array(M == 0)] = 0.0
    P = normalize_p(P)
    
    not_full = np.array(M > 0).astype(np.bool)
    sum_notfull = np.sum(not_full)
    
    if np.sum(M[not_full]) < t:
        raise ValueError("Cannot randomly pick number of required indices")
    
    result = []
    c = P.shape[0]
    for i in range(t):
        j = np.random.choice(c,p=P)
        
        assert not M[j] == 0
        M[j] -= 1
        if M[j] == 0:
            not_full[j] = False
            P[not_full] *= ((sum_notfull + P[j]) / sum_notfull)
            sum_notfull -= P[j]
            P[j] = 0
            P = normalize_p(P)
        
        result.append(j)
        
    return result
        
        
        
        



def apply_noise(Y,labeledIndexes,A,seed=None,deterministic=True):
    """ Corrupts a set percentage of initial labels with noise.
    
    Args:
        Y (`[NDArray[int].shape[N,C]`) : Matrix encoding initial beliefs.
        A (`[NDArray[int].shape[C,C]`): Transition probabilities between each class.
        labeledIndexes (`NDArray[bool].shape[N]`) : determines which indices are to be considered as labeled.
        seed (float) : Optional. Used to reproduce results. 
        
    Returns:
        `NDArray[int].shape[N,C]` : Belief matrix after corruption.
        
    """
    np.random.seed(seed)
    old_A = np.copy(np.asarray(A))
    if not np.all(old_A <= 1):
        print(old_A)
        raise Exception("trans. mat has value >1")
    old_Y = np.copy(Y)
    is_flat = np.ndim(Y) == 1
    if is_flat:
        Y = gutils.init_matrix(Y,labeledIndexes)
    c = Y.shape[1]
    n = Y.shape[0]
    
    Y = Y[labeledIndexes,:]
    Y_flat = np.argmax(Y,axis=1)

    vec = np.random.RandomState(seed).permutation(Y.shape[0])
    assert not vec is None
    cursor = np.zeros((c),dtype=np.int32)
    
    if deterministic == True:
        A = transition_count_mat(Y, A)
    else:
        
        
        
        
        class_freq = [int(np.sum(Y[:,i])) for i in range(c)]
        
        num_clean = np.sum(labeledIndexes) * sum([old_A[i,i] for i in range(c)])/c
        
        num_clean = int(np.round(num_clean))
        num_noisy = np.sum(labeledIndexes) - num_clean
        
        ##########3
        perm = np.random.permutation(Y.shape[0])[0:num_noisy]
        A = np.zeros((c,c))
        for i in range(c):
            A[i,i] = class_freq[i]
            
        for my_id in perm:
            j = np.argmax(Y[my_id,:])
            A[j,j] -= 1
            new_j = j
            while new_j == j:
                new_j = np.random.choice(c)
            A[j,new_j] += 1
        
        assert np.sum(A) == np.sum(labeledIndexes)
        print(A)
        ###############
       
        """
        
        B = __pick_from([class_freq[i] for i in range(c)], num_clean, M=[np.sum(Y[:,i]) for i in range(c)])
        
        
        A = np.zeros((c,c)).astype(np.int32)
        #print(B)
        for my_id in B:
            A[my_id,my_id] += 1
        assert np.sum(A) == num_clean
            
            
        #print(A)
        for row in range(c):
            row_clean = A[row,row]
            A[row,row] = 0
            M = np.inf * np.ones((c,))
            M[row] = 0
            B = __pick_from(old_A[row,:], class_freq[row]- row_clean, M=M)
            for my_id in B:
                A[row,my_id] += 1
            A[row,row] = row_clean
            assert np.sum(A[row,:]) == class_freq[row]
        assert np.sum(A) == np.sum(labeledIndexes)    
        
    print(A)
    raise ""
    """


    """
    if deterministic == False:
        for i in range(c):
            old_A[i,i] = 0.0
            if np.sum(old_A[i,:]) > 0:
                old_A[i,:] /= np.sum(old_A[i,:])
        
        for i in zip(range(A.shape[0])):
            num_clean = A[i,i]
            num_noisy = np.sum(A[i,:]) - num_clean 
            A[i,:] = 0
            A[i,i] = num_clean
            if num_noisy == 0:
                continue
            for noisy_choice in np.random.choice(c,num_noisy,p=np.squeeze(old_A[i,:]),replace=True):
                A[i,noisy_choice] += 1
    """           

    

    for i in np.arange(Y_flat.shape[0]):
        current_class = Y_flat[vec[i]]
        while A[current_class,cursor[current_class]] == 0:
            cursor[current_class] += 1
            assert cursor[current_class] < c
        Y_flat[vec[i]] = cursor[current_class]
        A[current_class,cursor[current_class]] -= 1
    
    
    noisy_Y = np.zeros(shape=(n,c))
    labeledIndexes_where = np.where(labeledIndexes)[0]
    for l in range(Y_flat.shape[0]):
        noisy_Y[labeledIndexes_where[l],Y_flat[l]] = 1    
    noisy_Y[np.logical_not(labeledIndexes),:] = old_Y[np.logical_not(labeledIndexes),:]
    print("Changed {} percent of entries".format(np.round(1-gutils.accuracy(np.argmax(Y,axis=1),Y_flat),6)))

    

 
    
    if is_flat:
        old_Y[labeledIndexes] = np.argmax(noisy_Y[labeledIndexes],axis=1)
        return old_Y
    else:
        return noisy_Y
    
    
    
if __name__ == "__main__":
    print(__pick_from([1.0,1.0,0.01], 10,M = [1,5,10]))