"""Orthogonal decomposition of features (parallel / orthogonal to a trivial subspace)."""
import torch


class Matrix(object):
    def __init__(self, vectors):
        self.vectors = vectors
        self.num_samples = vectors.shape[0]
        self.dimension = vectors.shape[1]

    def plus(self, v):
        return self.vectors + v.vectors

    def minus(self, v):
        return self.vectors - v.vectors

    def magnitude(self):
        return torch.sqrt(torch.sum(torch.pow(self.vectors, 2), dim=-1))

    def normalized(self):
        magnitude = self.magnitude().clamp(min=1e-8)
        weight = (1.0 / magnitude).reshape(self.num_samples, 1)
        return self.vectors * weight

    def component_parallel_to(self, basis):
        u = basis.normalized()
        weight = torch.sum(self.vectors * u, dim=-1).reshape(self.num_samples, 1)
        return u * weight

    def component_orthogonal_to(self, bais):
        projection = self.component_parallel_to(bais)
        return self.vectors - projection


def ortho_algorithm(original_feature, trivial_feature):
    """
    Returns the component of `original_feature` parallel to the direction orthogonal
    to `trivial_feature` (see original Orthographic algorithm).
    """
    original_feature = Matrix(original_feature)
    trivial_feature = Matrix(trivial_feature)
    d = original_feature.component_orthogonal_to(trivial_feature)
    d = Matrix(d)
    f = original_feature.component_parallel_to(d)
    return f.vectors
