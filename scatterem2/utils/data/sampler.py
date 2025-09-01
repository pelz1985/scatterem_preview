import time
from typing import Iterator, List, Sequence

import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import Sampler


class SparseSampler(Sampler):
    def __init__(
        self,
        indices: Sequence[int],
        positions: torch.Tensor,
        batch_size: int,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            indices (array-like): Indices of the dataset to sample from.
            positions (torch.Tensor): (N, 2) tensor of scan positions.
            batch_size (int): Desired batch size.
            verbose (bool): Print progress info.
        """
        self.indices: torch.Tensor = torch.tensor(indices, dtype=torch.long)
        self.positions: torch.Tensor = (
            positions
            if isinstance(positions, torch.Tensor)
            else torch.tensor(positions, dtype=torch.float32)
        )
        self.batch_size: int = batch_size
        self.verbose: bool = verbose
        self.groups: List[torch.Tensor] = self._make_sparse_groups()
        self.flattened: torch.Tensor = torch.cat(self.groups)

    def __iter__(self) -> Iterator[int]:
        # Yield all indices in sparse-batch order
        return iter(self.flattened.tolist())

    def __len__(self) -> int:
        return len(self.flattened)

    def _make_sparse_groups(self) -> List[torch.Tensor]:
        t0 = time.time()
        pos = self.positions
        indices = self.indices

        if len(indices) > len(pos):
            raise ValueError("More indices than positions.")
        if indices.max() >= len(pos):
            raise ValueError("Index exceeds position array.")

        num_batches: int = len(indices) // self.batch_size
        pos_s: torch.Tensor = pos[indices]

        # Clustering (MiniBatchKMeans) - convert to numpy for sklearn
        pos_s_np = pos_s.cpu().numpy()
        kmeans = MiniBatchKMeans(
            init="k-means++",
            n_init=10,
            n_clusters=num_batches,
            max_iter=10,
            batch_size=3072,
        )
        kmeans.fit(pos_s_np)
        labels: torch.Tensor = torch.tensor(kmeans.labels_, dtype=torch.long)

        # Group by cluster
        compact_batches: List[torch.Tensor] = [
            indices[torch.where(labels == i)[0]] for i in range(num_batches)
        ]

        # Compute centroids
        centroids: torch.Tensor = torch.stack(
            [pos[batch].float().mean(dim=0) for batch in compact_batches]
        )

        # Distance matrix - convert to numpy for scipy's cdist
        pos_np = pos.cpu().numpy()
        dist_matrix: torch.Tensor = torch.tensor(
            cdist(pos_np, pos_np), dtype=torch.float32
        )

        # Sparse init: closest point to each centroid
        sparse_batches: List[List[int]] = []
        sparse_indices: torch.Tensor = indices.clone()
        pos_s = pos[sparse_indices]
        used: List[int] = []

        for centroid in centroids:
            distances = torch.norm(pos_s - centroid, dim=1)
            closest_s = torch.argmin(distances)
            closest_idx = sparse_indices[closest_s]
            sparse_batches.append([closest_idx.item()])
            used.append(closest_s.item())

        # Remove used indices
        mask = torch.ones(len(sparse_indices), dtype=torch.bool)
        mask[used] = False
        sparse_indices = sparse_indices[mask]

        # Greedy max-min assignment
        for idx in sparse_indices:
            dists: List[float] = []
            for j in range(num_batches):
                batch_tensor = torch.tensor(sparse_batches[j], dtype=torch.long)
                min_dist = torch.min(dist_matrix[batch_tensor, idx])
                dists.append(min_dist.item())

            best_batch = torch.argmax(torch.tensor(dists))
            sparse_batches[best_batch.item()].append(idx.item())

        # Convert to tensors
        sparse_batches_tensor: List[torch.Tensor] = [
            torch.tensor(batch, dtype=torch.long) for batch in sparse_batches
        ]

        # Sanity check
        flat = torch.cat(sparse_batches_tensor)
        assert torch.equal(
            torch.sort(flat)[0], torch.sort(indices)[0]
        ), "Mismatch in sparse batching."

        if self.verbose:
            print(
                f"[SparseSampler] Generated {num_batches} sparse groups in {time.time() - t0:.2f}s"
            )

        return sparse_batches_tensor
