import numpy as np
import torch


def rand_projections(embedding_dim, num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)


def dis_swd(dist0, dist1, num_projections=50, p=2, device='0'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    embedding_dim = dist0.size(1)
    projections = rand_projections(embedding_dim, num_projections).to(device)
    dist0_projections = dist0.matmul(projections.transpose(0, 1))
    dist1_projections = dist1.matmul(projections.transpose(0, 1))
    wasserstein_distance = (torch.sort(dist0_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(dist1_projections.transpose(0, 1), dim=1)[0])
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    return wasserstein_distance.mean()


def dis_swd_numpy(dist0, dist1, num_projections=50, p=2):
    embedding_dim = dist0.shape[1]
    projections = rand_projections(embedding_dim, num_projections)
    projections = np.transpose(projections)
    dist0_projections = np.dot(dist0, projections)
    dist1_projections = np.dot(dist1, projections)
    wasserstein_distance = np.sort(dist0_projections, axis=0) - np.sort(dist1_projections, axis=0)
    wasserstein_distance = np.power(wasserstein_distance, p)
    swd = np.mean(wasserstein_distance)
    return swd


def dis_swd_torch(dist0, dist1, num_projections=50, p=2, device='cuda'):
    embedding_dim = dist0.size(1)
    # w = torch.normal(mean=0.0, std=1.0, size=(num_projections, embedding_dim), device='cuda')
    # for n in range(num_projections):
    #     div = float(torch.sqrt((w[n]**2).sum()))
    #     w[n] = w[n] / div

    # normals = torch.randn(sizes=(num_projections, embedding_dim))
    projections = rand_projections(embedding_dim, num_projections).to(device)
    dist0_projections = dist0.matmul(projections.transpose(0, 1))
    dist1_projections = dist1.matmul(projections.transpose(0, 1))
    wasserstein_distance = (torch.sort(dist0_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(dist1_projections.transpose(0, 1), dim=1)[0])
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    return wasserstein_distance.mean()


if __name__ == '__main__':
    # dist0 = np.random.random(size=(32, 16))
    # dist1 = np.random.uniform(size=(32, 16))
    # swd = dis_swd_numpy(dist0, dist1)
    dist0 = torch.randn(32, 16, 5, 5).cuda()
    dist1 = torch.randn(32, 16, 5, 5).cuda()
    swd = dis_swd_torch(dist0, dist1)
    print(dist0[:2])
    print(dist1[:2])
    print(swd)
