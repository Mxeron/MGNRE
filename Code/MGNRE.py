import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from GB_generation_with_idx import get_GB
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def NR_torch(X_temp, radius):
    n, m = X_temp.shape
    if m == 0:
        return torch.ones((n, n), device=X_temp.device, dtype=torch.bool)
        
    dist_matrix = torch.cdist(X_temp, X_temp, p=2) / m
    return dist_matrix <= radius

def MGNRE(X, sigma):
    X_torch = torch.from_numpy(X).float().to(device)
    n, m = X_torch.shape
    Imp = torch.zeros(m, device=device)
    NRs = torch.zeros((m, n, n), device=device)
    
    for k in range(m):
        radius = torch.std(X_torch[:, k]) / sigma
        temp = NR_torch(X_torch[:, [k]], radius)
        NRs[k] = temp.float() + 1e-4
        cardinalities = torch.sum(NRs[k], axis=1)
        Imp[k] = -torch.mean(torch.log2(cardinalities / n))
        
    b_as = torch.argsort(Imp)
    weight_as = torch.zeros((n, m), device=device)
    weight_single = torch.zeros((n, m), device=device)
    NE_as = torch.zeros(m, device=device)
    NE_single = torch.zeros(m, device=device)
    NE_as_x = torch.zeros((n, m), device=device)
    NE_single_x = torch.zeros((n, m), device=device)
    rnc_as = torch.zeros((n, m), device=device)
    rnc_single = torch.zeros((n, m), device=device)
    
    for k in range(m):
        as_indices = b_as[:m - k]
        
        sub_X = X_torch[:, as_indices]
        radius = torch.std(sub_X) / sigma
        NR_as = NR_torch(sub_X, radius).float() + 1e-4
        
        NR_single = NRs[k]

        weight_as[:, k] = torch.sqrt(torch.sum(NR_as, axis=1) / n)
        weight_single[:, k] = torch.sqrt(torch.sum(NR_single, axis=1) / n)
        
        NE_as[k] = -torch.mean(torch.log2(torch.sum(NR_as, axis=1) / n))
        NE_single[k] = -torch.mean(torch.log2(torch.sum(NR_single, axis=1) / n))
        
        sum_full_as = torch.sum(NR_as)
        row_sums_as = torch.sum(NR_as, axis=1)
        sum_after_delete_as = sum_full_as - 2 * row_sums_as + torch.diag(NR_as)
        rnc_as[:, k] = row_sums_as - sum_after_delete_as / (n - 1)

        sum_full_single = torch.sum(NR_single)
        row_sums_single = torch.sum(NR_single, axis=1)
        sum_after_delete_single = sum_full_single - 2 * row_sums_single + torch.diag(NR_single)
        rnc_single[:, k] = row_sums_single - sum_after_delete_single / (n - 1)
        
        fc_full_as = torch.sum(NR_as, axis=1)
        fc_full_single = torch.sum(NR_single, axis=1)
        cur_x_n = n - 1
        for i in range(n):
            fc_sub_as = fc_full_as - NR_as[:, i]
            fc_sub_deleted_as = torch.cat((fc_sub_as[:i], fc_sub_as[i+1:]))
            NE_as_x[i, k] = -torch.mean(torch.log2(fc_sub_deleted_as / cur_x_n))
            
            fc_sub_single = fc_full_single - NR_single[:, i]
            fc_sub_deleted_single = torch.cat((fc_sub_single[:i], fc_sub_single[i+1:]))
            NE_single_x[i, k] = -torch.mean(torch.log2(fc_sub_deleted_single / cur_x_n))

    rne_x_as = 1 - NE_as_x / (NE_as)
    rne_x_as = torch.clamp(rne_x_as, 0, 1)
    
    nod_as_pos = rne_x_as * (n - torch.abs(rnc_as)) / (2 * n)
    nod_as_neg = rne_x_as * torch.sqrt((n + torch.abs(rnc_as)) / (2 * n))
    nod_as = torch.where(rnc_as > 0, nod_as_pos, nod_as_neg)
    
    rne_x_single = 1 - NE_single_x / (NE_single)
    rne_x_single = torch.clamp(rne_x_single, 0, 1)
    
    nod_single_pos = rne_x_single * (n - torch.abs(rnc_single)) / (2 * n)
    nod_single_neg = rne_x_single * torch.sqrt((n + torch.abs(rnc_single)) / (2 * n))
    nod_single = torch.where(rnc_single > 0, nod_single_pos, nod_single_neg)

    sum_single = torch.sum((1 - nod_single) * weight_single, axis=1)
    sum_as = torch.sum((1 - nod_as) * weight_as, axis=1)
    OS = 1 - (sum_single + sum_as) / (2 * m)
    
    return OS.cpu().numpy()


if __name__ == "__main__":
    # data_path = "./Datasets/lymphography.mat"
    data_path = "./lymphography.mat"
    load_data = loadmat(data_path)
    trandata = load_data['trandata']
    X = trandata[:,:-1]
    labels = trandata[:,-1]
    n, m = X.shape
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    GBs = get_GB(X)
    n_gb = len(GBs)
    centers = np.zeros((n_gb, m))
    for idx, gb in enumerate(GBs):
        centers[idx] = np.mean(gb[:,:-1], axis=0)
        
    GBOS = MGNRE(centers, 1.6)
    OS = np.zeros(n)
    for idx, gb in enumerate(GBs):
        point_idxs = gb[:,-1].astype('int')
        OS[point_idxs] = GBOS[idx]
    print(OS)