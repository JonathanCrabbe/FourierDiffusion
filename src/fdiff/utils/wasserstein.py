""" Calculate Wasserstein distances between two datasets.
    Code addapted form https://gitlab.developers.cam.ac.uk/maths/cia/covid-19-projects/missing_data_fitting_quality
"""

from typing import Optional

import numpy as np
import ot
from tqdm import tqdm


class WassersteinDistances:
    """Calculate Wasserstein distance of two datasets in various ways.

    Parameters
    ----------
    original_data : np.ndarray
        Original data set, an (n, d) ndarray.
    other_data : np.ndarray
        Other data set, which might be imputed or simulated data, also
        an (n, d) ndarray.
    normalisation : Normalisation
        Method of normalising data.  If 'none', no normalisation will be used.
        If 'standatdise', then standardise the data by dividing by the
        standard deviation of the original data.  (There is no need to
        subtract the mean, as this does not affect the Wasserstein distance.)

    """

    def __init__(
        self,
        original_data: np.ndarray,
        other_data: np.ndarray,
        normalisation: Optional[str] = "none",
        seed: Optional[int] = None,
    ) -> None:
        self.original_data = original_data
        self.other_data = other_data
        self.normalisation = normalisation
        self.rng = np.random.default_rng(seed)

    def random_direction(self, dim: int) -> np.ndarray:
        """Generate a unit vector in a random direction.

        Parameters
        ----------
        dim : int
            Dimension of vector to be generated.

        Returns
        -------
        unit_vector : np.ndarray
            A unit vector of shape (dim,).

        """
        vector = self.rng.normal(size=dim)
        vector_magnitude = np.linalg.norm(vector)
        unit_vector = vector / vector_magnitude
        return unit_vector

    def get_random_directions(self, n_directions: int) -> list[np.ndarray]:
        """Get random directions for an experiment.

        Parameters
        ----------
        n_directions : int
            The number of directions to produce.

        Returns
        -------
        directions : list[np.ndarray]
            A list of unit vectors specifying the directions to use.  The
            results will be given in the same order.

        """
        dimension = self.original_data.shape[1]
        directions = [self.random_direction(dimension) for _ in range(n_directions)]
        return directions

    def get_marginal_directions(self) -> list[np.ndarray]:
        """Get marginal directions for an experiment.

        These are just the standard basis vectors.

        Returns
        -------
        directions : list[np.ndarray]
            A list of standard unit vectors.

        """
        dimension = self.original_data.shape[1]
        directions = [np.identity(dimension)[i] for i in range(dimension)]
        return directions

    def feature_distance(self, feature: int) -> float:
        """Calculate the dataset distance for a specific feature.

        This calculates the Wasserstein 2-distance between the
        specified feature in the two datasets.

        Parameters
        ----------
        feature : int
            The column number of the feature to consider: 0, 1, 2, ...,
            `num_fields` - 1.

        Returns
        -------
        distance : float
            The Wasserstein 2-distance.

        """
        original = self.original_data[:, feature]
        other = self.other_data[:, feature]
        original_normalised, other_normalised = self._normalise(original, other)
        distance = ot.emd2_1d(original_normalised, other_normalised)
        distance = np.sqrt(distance)
        return float(distance)

    def directional_distance(self, direction: np.ndarray) -> float:
        """Calculate the dataset distance in a specified direction.

        This projects the two datasets onto the specified direction (that is,
        a 1-dimensional subspace), and calculates the Wasserstein distance
        between the two resulting distributions.

        Parameters
        ----------
        direction : np.array
            The direction in which to calculate the W_2 distance between
            the datasets.

        Returns
        -------
        distance : float
            The calculated W_2^2 distance.

        """
        original = self._project(self.original_data, direction)
        other = self._project(self.other_data, direction)
        original_normed, other_normed = self._normalise(original, other)
        distance = ot.emd2_1d(original_normed, other_normed)
        distance = np.sqrt(distance)
        return float(distance)

    @staticmethod
    def _project(data: np.ndarray, direction: np.ndarray) -> np.ndarray:
        proj = data @ direction
        assert isinstance(proj, np.ndarray)
        return proj

    def _normalise(
        self, orig: np.ndarray, other: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.normalisation == "none":
            return orig, other
        if self.normalisation == "standardise":
            sd = np.std(orig)
            return orig / sd, other / sd
        raise ValueError(f"Unrecognised normalisation type: {self.normalisation}")

    def sliced_distances(self, num_directions: int) -> np.ndarray:
        """Calculate the sliced Wasserstein distance between datasets.

        Args:
            num_directions (int): Number of directions in the sliced Wasserstein estimation.

        Returns:
            np.ndarray: distribution of Wasserstein distances over all directions.
        """
        directions = self.get_random_directions(num_directions)
        distances = []
        for direction in tqdm(
            directions,
            desc="Computing sliced Wasserstein",
            unit="proj",
            leave=False,
            colour="blue",
        ):
            distances.append(self.directional_distance(direction))
        return np.array(distances)

    def marginal_distances(self) -> np.ndarray:
        """Calculate the marginal Wasserstein distances between datasets.

        Returns:
            np.ndarray: distribution of Wasserstein distances over all features.
        """
        n_features = self.original_data.shape[1]
        distances = []
        for feature in tqdm(
            range(n_features),
            desc="Computing marginal Wasserstein",
            unit="feature",
            leave=False,
            colour="blue",
        ):
            distances.append(self.feature_distance(feature))
        return np.array(distances)
