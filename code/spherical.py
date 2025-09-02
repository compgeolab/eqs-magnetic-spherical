"""
Implements the forward modeling and equivalent sources

Also includes some utilities for general use.
"""

import numpy as np
import numba
import pyproj
import rich.progress
import choclo
import verde as vd
import verde.base
import harmonica as hm
import bordado as bd
import boule as bl


CM = choclo.constants.VACUUM_MAGNETIC_PERMEABILITY / 4 / np.pi


def vector_geodetic_to_spherical(
    eastward,
    northward,
    upward,
    latitude_spherical,
    latitude,
):
    """
    Rotate a vector from a geodetic to a spherical system.
    """
    angle = np.radians(latitude - latitude_spherical)
    cos = np.cos(angle)
    sin = np.sin(angle)
    # Multiply the vector by a rotation matrix around the eastward direction
    northward_spherical = cos * northward + sin * upward
    radial = -sin * northward + cos * upward
    return eastward, northward_spherical, radial


def vector_spherical_to_geodetic(
    eastward,
    northward,
    radial,
    latitude_spherical,
    latitude,
):
    """
    Rotate a vector from a spherical to a geodetic system.
    """
    return vector_geodetic_to_spherical(
        eastward,
        northward,
        radial,
        latitude,
        latitude_spherical,
    )


def dipole_magnetic_geodetic(coordinates, dipoles, magnetic_moments):
    """
    Calculate the magnetic field of dipoles in geodetic coordinates.

    Input coordinates are geodetic and the output vector is in
    a local north-oriented Cartesian frame with the z-axis aligned with the
    direction normal to the ellipsoid.

    Parameters
    ----------
    coordinates : tuple = (longitude, latitude, height)
        Tuple of arrays containing the longitude, latitude, and height
        coordinates of the computation points defined on a geodetic coordinate
        system. Longitude and latitude should be in decimal degrees and height
        should be in meters. Coordinates can be n-dimensional arrays but all
        arrays must have the same shape.
    dipoles: tuple = (longitude, latitude, height)
        Tuple of arrays containing the longitude, latitude, and height
        coordinates of the dipoles defined on a geodetic coordinate system.
        Longitude and latitude should be in decimal degrees and height should
        be in meters. Coordinates can be n-dimensional arrays but all arrays
        must have the same shape.
    magnetic_moments: tuple = (east, north, up)
        Tuple of arrays containing the magnetic moments of the dipoles, given
        by three components: longitudinal, latitudinal, and upward. Each
        component should be an n-dimensional array with the same shape as the
        dipole coordinates. The vector components are defined on a local
        north-oriented Cartesian system located at the dipole.

    Returns
    -------
    magnetic_field : tuple = (east, north, up)
        Arrays with the longitudinal, latitudinal, and upward components of the
        combined B magnetic field of the dipoles in nT.

    """
    # Convert coordinates and dipole moments to spherical
    coordinates_sph = bl.WGS84.geodetic_to_spherical(
        coordinates[0], coordinates[1], coordinates[2]
    )
    dipoles_sph = bl.WGS84.geodetic_to_spherical(dipoles[0], dipoles[1], dipoles[2])
    magnetic_moments_sph = vector_geodetic_to_spherical(
        eastward=magnetic_moments[0],
        northward=magnetic_moments[1],
        upward=magnetic_moments[2],
        latitude_spherical=dipoles_sph[1],
        latitude=dipoles[1],
    )
    b_lon, b_lat, b_radial = dipole_magnetic_spherical(
        coordinates_sph, dipoles_sph, magnetic_moments_sph
    )
    b_lon, b_lat, b_height = vector_spherical_to_geodetic(
        eastward=b_lon,
        northward=b_lat,
        radial=b_radial,
        latitude_spherical=coordinates_sph[1],
        latitude=coordinates[1],
    )
    return b_lon, b_lat, b_height


def dipole_magnetic_spherical(coordinates, dipoles, magnetic_moments):
    """
    Calculate the magnetic field of dipoles in spherical coordinates

    Input coordinates are geocentric spherical and the output vector is in
    a local north-oriented Cartesian frame with the z-axis aligned with the
    radial direction.

    Parameters
    ----------
    coordinates : tuple = (longitude, latitude, radius)
        Tuple of arrays containing the longitude, spherical latitude, and
        radius coordinates of the computation points defined on a geocentric
        spherical coordinate system. Longitude and latitude should be in
        decimal degrees and radius should be in meters. Coordinates can be
        n-dimensional arrays but all arrays must have the same shape.
    dipoles: tuple = (longitude, latitude, radius)
        Tuple of arrays containing the longitude, spherical latitude, and
        radius coordinates of the dipoles defined on a geocentric spherical
        coordinate system. Longitude and latitude should be in decimal degrees
        and radius should be in meters. Coordinates can be n-dimensional arrays
        but all arrays must have the same shape.
    magnetic_moments: tuple = (east, north, radial)
        Tuple of arrays containing the magnetic moments of the dipoles, given
        by three components: longitudinal, latitudinal, and radial. Each
        component should be an n-dimensional array with the same shape as the
        dipole coordinates. The vector components are defined on a local
        north-oriented Cartesian system located at the dipole.

    Returns
    -------
    magnetic_field : tuple = (east, north, radial)
        Arrays with the longitudinal, latitudinal, and radial components of the
        combined B magnetic field of the dipoles in nT.

    Notes
    -----
    This function:
    1. Converts latitude and longitude from degrees to radians for spherical
       coordinate calculations.
    2. Calculates directional cosines and separation distances between source
       and observation points.
    3. Computes intermediate matrices (H_ij) accounting for the geometry and
       separation between source and observation points.
    4. Uses the magnetic moments and geometry to compute the magnetic field
       components in spherical coordinates.
    """
    # Convert to 1D arrays to make it easier to loop over them
    shape = coordinates[0].shape
    n_data = coordinates[0].size
    coordinates = tuple(np.atleast_1d(c).ravel() for c in coordinates)
    dipoles = tuple(np.atleast_1d(c).ravel() for c in dipoles)
    magnetic_moments = tuple(np.atleast_1d(c).ravel() for c in magnetic_moments)
    # Initialize the output arrays
    b_lon = np.zeros(n_data)
    b_lat = np.zeros(n_data)
    b_radial = np.zeros(n_data)
    # This function needs the colatitude and all angles in radians
    _dipole_magnetic_spherical_fast(
        np.radians(coordinates[0]),
        np.radians(90 - coordinates[1]),
        coordinates[2],
        np.radians(dipoles[0]),
        np.radians(90 - dipoles[1]),
        dipoles[2],
        magnetic_moments[0],
        -magnetic_moments[1],
        magnetic_moments[2],
        b_lon,
        b_lat,
        b_radial,
    )
    # Reshape things back to the original coordinate shapes
    b_lon = b_lon.reshape(shape)
    b_lat = b_lat.reshape(shape)
    b_radial = b_radial.reshape(shape)
    return b_lon, b_lat, b_radial


@numba.jit(nopython=True, parallel=True)
def _dipole_magnetic_spherical_fast(
    longitude,
    colatitude,
    radius,
    longitude_d,
    colatitude_d,
    radius_d,
    m_lon,
    m_colat,
    m_radial,
    b_lon,
    b_lat,
    b_radial,
):
    n_dipoles = longitude_d.size
    n_data = longitude.size
    for j in numba.prange(n_dipoles):
        sin_colat_d = np.sin(colatitude_d[j])
        cos_colat_d = np.cos(colatitude_d[j])
        for i in range(n_data):
            sin_colat = np.sin(colatitude[i])
            cos_colat = np.cos(colatitude[i])
            sin_lon = np.sin(longitude[i] - longitude_d[j])
            cos_lon = np.cos(longitude[i] - longitude_d[j])
            b_lon_j, b_lat_j, b_radial_j = _kernel(
                cos_lon,
                sin_lon,
                cos_colat,
                sin_colat,
                radius[i],
                cos_colat_d,
                sin_colat_d,
                radius_d[j],
                m_lon[j],
                m_colat[j],
                m_radial[j],
            )
            b_lon[i] += b_lon_j
            b_lat[i] += b_lat_j
            b_radial[i] += b_radial_j


@numba.jit(nopython=True)
def _kernel(
    cos_lon,
    sin_lon,
    cos_colat,
    sin_colat,
    radius,
    cos_colat_d,
    sin_colat_d,
    radius_d,
    m_lon,
    m_colat,
    m_radial,
):
    """
    Just the forward modeling kernel equations.
    """
    mu_ij = cos_colat * cos_colat_d + sin_colat * sin_colat_d * cos_lon
    ri_dot_thetaj = -cos_colat * sin_colat_d + sin_colat * cos_colat_d * cos_lon
    ri_dot_phij = sin_colat * sin_lon
    thetai_dot_rj = -sin_colat * cos_colat_d + cos_colat * sin_colat_d * cos_lon
    thetai_dot_thetaj = sin_colat * sin_colat_d + cos_colat * cos_colat_d * cos_lon
    thetai_dot_phij = cos_colat * sin_lon
    phii_dot_rj = -sin_colat_d * sin_lon
    phii_dot_thetaj = -cos_colat_d * sin_lon
    phii_dot_phij = cos_lon

    # Distance r_ij between the computation point and the dipole
    r_ij = np.sqrt((radius**2) + (radius_d**2) - 2 * radius * radius_d * mu_ij)
    r_ij2 = r_ij**2

    # Define magnetic field terms
    CM_rij3 = CM / r_ij**3
    H_11 = CM_rij3 * (
        3 * ((radius - radius_d * mu_ij) * (radius * mu_ij - radius_d) / r_ij2) - mu_ij
    )
    H_12 = CM_rij3 * (
        3 * ((radius - radius_d * mu_ij) * (radius * ri_dot_thetaj) / r_ij2)
        - ri_dot_thetaj
    )
    H_13 = CM_rij3 * (
        3 * ((radius - radius_d * mu_ij) * (radius * ri_dot_phij) / r_ij2) - ri_dot_phij
    )
    H_21 = -CM_rij3 * (
        3 * ((radius_d * thetai_dot_rj) * (radius * mu_ij - radius_d) / r_ij2)
        + thetai_dot_rj
    )
    H_22 = -CM_rij3 * (
        3 * ((radius_d * thetai_dot_rj) * (radius * ri_dot_thetaj) / r_ij2)
        + thetai_dot_thetaj
    )
    H_23 = -CM_rij3 * (
        3 * ((radius_d * thetai_dot_rj) * (radius * ri_dot_phij) / r_ij2)
        + thetai_dot_phij
    )
    H_31 = -CM_rij3 * (
        3 * ((radius_d * phii_dot_rj) * (radius * mu_ij - radius_d) / r_ij2)
        + phii_dot_rj
    )
    H_32 = -CM_rij3 * (
        3 * ((radius_d * phii_dot_rj) * (radius * ri_dot_thetaj) / r_ij2)
        + phii_dot_thetaj
    )
    H_33 = -CM_rij3 * (
        3 * ((radius_d * phii_dot_rj * radius * ri_dot_phij) / r_ij2) + phii_dot_phij
    )

    # Convert the magnetic moment to the local Cartesian system of the
    # observation point P.
    m_radial_p = m_radial * mu_ij + m_colat * ri_dot_thetaj + m_lon * ri_dot_phij
    m_colat_p = (
        m_radial * thetai_dot_rj + m_colat * thetai_dot_thetaj + m_lon * thetai_dot_phij
    )
    m_lon_p = m_radial * phii_dot_rj + m_colat * phii_dot_thetaj + m_lon * phii_dot_phij

    # Calculate final magnetic field components and convert them from T to nT
    b_lon = (H_31 * m_radial_p + H_32 * m_colat_p + H_33 * m_lon_p) * 1e9
    # The - converts colatitude to latitude.
    b_lat = -(H_21 * m_radial_p + H_22 * m_colat_p + H_23 * m_lon_p) * 1e9
    b_radial = (H_11 * m_radial_p + H_12 * m_colat_p + H_13 * m_lon_p) * 1e9

    return b_lon, b_lat, b_radial


@numba.jit(nopython=True, parallel=True)
def _jacobian_fast(
    longitude,
    colatitude,
    radius,
    longitude_source,
    colatitude_source,
    radius_source,
    moment_east,
    moment_north,
    moment_radial,
    main_field_east,
    main_field_north,
    main_field_radial,
    result,
):
    n_dipoles = longitude_source.size
    n_data = longitude.size
    for j in numba.prange(n_dipoles):
        sin_colat_d = np.sin(colatitude_source[j])
        cos_colat_d = np.cos(colatitude_source[j])
        for i in range(n_data):
            sin_colat = np.sin(colatitude[i])
            cos_colat = np.cos(colatitude[i])
            sin_lon = np.sin(longitude[i] - longitude_source[j])
            cos_lon = np.cos(longitude[i] - longitude_source[j])
            b_east, b_north, b_radial = _kernel(
                cos_lon,
                sin_lon,
                cos_colat,
                sin_colat,
                radius[i],
                cos_colat_d,
                sin_colat_d,
                radius_source[j],
                moment_east[j],
                -moment_north[j],
                moment_radial[j],
            )
            result[i, j] = (
                main_field_east[i] * b_east
                + main_field_north[i] * b_north
                + main_field_radial[i] * b_radial
            )


class EquivalentSourcesMagGeod:
    """
    Magnetic equivalent sources in geodetic coordinates.
    """

    def __init__(
        self,
        damping=None,
        depth=None,
        inclination=90,
        declination=0,
        source_coordinates=None,
        ellipsoid=bl.WGS84,
    ):
        self.damping = damping
        self.depth = depth
        self.inclination = inclination
        self.declination = declination
        self.source_coordinates = source_coordinates
        self.ellipsoid = ellipsoid

    def predict(self, coordinates):
        if not hasattr(self, "dipole_moments_"):
            raise ValueError("Fit the class before predicting.")
        result = dipole_magnetic_geodetic(
            coordinates, self.source_coordinates_, self.dipole_moments_
        )
        return result

    def _estimate_depth(self, coordinates):
        """
        Estimate a reasonable depth if one isn't given.
        """

        coslat = np.cos(np.radians(coordinates[1]))
        sinlat = np.sin(np.radians(coordinates[1]))
        N = self.ellipsoid.prime_vertical_radius(sinlat)
        b = self.ellipsoid.semiminor_axis
        a = self.ellipsoid.semimajor_axis
        coordinates_cartesian = (
            (N + coordinates[2]) * coslat * np.cos(np.radians(coordinates[0])),
            (N + coordinates[2]) * coslat * np.sin(np.radians(coordinates[0])),
            (b**2 * N / a**2 + coordinates[2]) * sinlat,
        )
        return 5 * bd.neighbor_distance_statistics(
            coordinates_cartesian, "median", k=10
        )

    def fit(self, coordinates, inclination, declination, data, weights=None):
        """ """
        # Ravel all arrays because the Jacobian calculation needs them ravelled
        coordinates = tuple(np.atleast_1d(c).ravel() for c in coordinates)
        data = data.ravel()
        # Calculate a default depth if needed
        if self.depth is None:
            self.depth_ = self._estimate_depth(coordinates)
        else:
            self.depth_ = self.depth
        # If source coordinates aren't given, use the data coordinates
        if self.source_coordinates is None:
            self.source_coordinates_ = (
                coordinates[0].copy(),
                coordinates[1].copy(),
                coordinates[2] - self.depth_,
            )
        else:
            self.source_coordinates_ = self.source_coordinates
        # Convert everything to a spherical coordinate system
        coordinates_sph = bl.WGS84.geodetic_to_spherical(*coordinates)
        source_coordinates_sph = bl.WGS84.geodetic_to_spherical(
            *self.source_coordinates_
        )
        n_data = coordinates[0].size
        n_params = source_coordinates_sph[0].size
        unit_moment = vector_geodetic_to_spherical(
            *hm.magnetic_angles_to_vec(
                np.ones(n_params), self.inclination, self.declination
            ),
            latitude_spherical=source_coordinates_sph[1],
            latitude=self.source_coordinates_[1],
        )
        main_field = vector_geodetic_to_spherical(
            *hm.magnetic_angles_to_vec(np.ones(n_data), inclination, declination),
            latitude_spherical=coordinates_sph[1],
            latitude=coordinates[1],
        )
        # Solve the inverse problem
        jacobian = self._jacobian(
            coordinates_sph, source_coordinates_sph, unit_moment, main_field
        )
        moment_amplitude = verde.base.least_squares(
            jacobian, data, weights=weights, damping=self.damping
        )
        # Calculate moment vectors and store them
        self.dipole_moments_ = hm.magnetic_angles_to_vec(
            moment_amplitude, self.inclination, self.declination
        )
        return self

    def _jacobian(
        self, coordinates_sph, source_coordinates_sph, unit_moment, main_field
    ):
        """ """
        n_data = coordinates_sph[0].size
        n_params = source_coordinates_sph[0].size
        jacobian = np.empty((n_data, n_params))
        _jacobian_fast(
            longitude=np.radians(coordinates_sph[0]),
            colatitude=np.radians(90 - coordinates_sph[1]),
            radius=coordinates_sph[2],
            longitude_source=np.radians(source_coordinates_sph[0]),
            colatitude_source=np.radians(90 - source_coordinates_sph[1]),
            radius_source=source_coordinates_sph[2],
            moment_east=unit_moment[0],
            moment_north=unit_moment[1],
            moment_radial=unit_moment[2],
            main_field_east=main_field[0],
            main_field_north=main_field[1],
            main_field_radial=main_field[2],
            result=jacobian,
        )
        return jacobian


class EquivalentSourcesMagGeodGB(EquivalentSourcesMagGeod):
    """
    Magnetic gradient-boosted equivalent sources in geodetic coordinates.

    Uses a Lambert Cylindrical Equal Area projection to define the windows,
    avoiding issues with convergence of longitude lines towards the poles.
    Window size is specified in meters.
    """

    def __init__(
        self,
        damping=None,
        depth=None,
        inclination=90,
        declination=0,
        source_coordinates=None,
        window_size=None,
        n_points_per_window=5e3,
        random_seed=None,
        verbose=True,
    ):
        super().__init__(
            damping=damping,
            depth=depth,
            inclination=inclination,
            declination=declination,
            source_coordinates=source_coordinates,
        )
        self.window_size = window_size
        self.n_points_per_window = n_points_per_window
        self.random_seed = random_seed
        self.verbose = verbose

    def fit(self, coordinates, inclination, declination, data, weights=None):
        """ """
        # Ravel all arrays because the Jacobian calculation needs them ravelled
        coordinates = tuple(np.atleast_1d(c).ravel() for c in coordinates)
        data = data.ravel()
        # Calculate a default depth if needed
        if self.depth is None:
            self.depth_ = self._estimate_depth(coordinates)
        else:
            self.depth_ = self.depth
        # If source coordinates aren't given, use the data coordinates
        if self.source_coordinates is None:
            self.source_coordinates_ = (
                coordinates[0].copy(),
                coordinates[1].copy(),
                coordinates[2] - self.depth_,
            )
        else:
            self.source_coordinates_ = self.source_coordinates
        n_data = coordinates[0].size
        n_params = self.source_coordinates_[0].size
        # Define a default window size if one is not given
        if self.window_size is None:
            # Estimate a latitude window size (in degrees) so each window has ~5k points
            region = bd.get_region(coordinates[:2])
            dlon = np.radians(region[1] - region[0])
            R = 6371.0
            area = (R**2) * dlon * (np.sin(np.radians(region[3])) - np.sin(np.radians(region[2])))
            points_per_km2 = n_data / area
            window_area = self.n_points_per_window / points_per_km2
            window_size_km = np.sqrt(window_area)
            # Convert the window size back to degrees
            self.window_size_ = np.degrees(window_size_km / R)
            min_region_dim = min([
                coordinates[1].max() - coordinates[1].min(),  # latitude extent
                coordinates[0].max() - coordinates[0].min(),  # longitude extent
            ])

            if self.window_size_ > min_region_dim:
                self.window_size_ = min_region_dim
        else:
            self.window_size_ = self.window_size

        # Convert everything to a spherical coordinate system
        coordinates_sph = bl.WGS84.geodetic_to_spherical(*coordinates)
        source_coordinates_sph = bl.WGS84.geodetic_to_spherical(
            *self.source_coordinates_
        )
        unit_moment = vector_geodetic_to_spherical(
            *hm.magnetic_angles_to_vec(
                np.ones(n_params), self.inclination, self.declination
            ),
            latitude_spherical=source_coordinates_sph[1],
            latitude=self.source_coordinates_[1],
        )
        main_field = vector_geodetic_to_spherical(
            *hm.magnetic_angles_to_vec(np.ones(n_data), inclination, declination),
            latitude_spherical=coordinates_sph[1],
            latitude=coordinates[1],
        )
        # Create the window indices directly on the sphere
        region_geo = bd.get_region(coordinates[:2])
        window_centers_geo, data_indices = bd.rolling_window_spherical(
            coordinates[:2],
            self.window_size_,
            overlap=0.5,
            region=region_geo,
        )
        _, source_indices = bd.rolling_window_spherical(
            self.source_coordinates_[:2],
            self.window_size_,
            overlap=0.5,
            region=region_geo,
        )
        self.window_centers_ = window_centers_geo
        source_indices = source_indices.ravel()
        data_indices = data_indices.ravel()
        # Initialize the solution
        self.residuals_ = data.copy()
        moment_amplitude = np.zeros(n_params)
        # Gradient-boosting iterations
        window_indices = list(range(data_indices.size))
        np.random.default_rng(self.random_seed).shuffle(window_indices)
        if self.verbose:
            window_indices = rich.progress.track(window_indices)
        for i in window_indices:
            # Skip windows with no data in them
            if data_indices[i][0].size == 0:
                continue
            # Select data and sources inside the window
            coordinates_window = tuple(c[data_indices[i]] for c in coordinates_sph)
            main_field_window = tuple(c[data_indices[i]] for c in main_field)
            if weights is not None:
                weights_window = tuple(c[data_indices[i]] for c in weights)
            else:
                weights_window = None
            source_coordinates_window = tuple(
                c[source_indices[i]] for c in source_coordinates_sph
            )
            unit_moment_window = tuple(c[source_indices[i]] for c in unit_moment)
            # Solve the inverse problem
            jacobian = self._jacobian(
                coordinates_window,
                source_coordinates_window,
                unit_moment_window,
                main_field_window,
            )
            moment_amplitude_window = verde.base.least_squares(
                jacobian,
                self.residuals_[data_indices[i]],
                weights=weights_window,
                damping=self.damping,
            )
            moment_amplitude[source_indices[i]] += moment_amplitude_window
            # Use the unit vectors to avoid geodetic to spherical conversions
            dipole_moment_window = tuple(
                c * moment_amplitude_window for c in unit_moment_window
            )
            predicted_field = dipole_magnetic_spherical(
                coordinates_sph, source_coordinates_window, dipole_moment_window
            )
            self.residuals_ -= sum(f * b for f, b in zip(main_field, predicted_field))
        # Calculate moment vectors and store them
        self.dipole_moments_ = hm.magnetic_angles_to_vec(
            moment_amplitude, self.inclination, self.declination
        )
        return self