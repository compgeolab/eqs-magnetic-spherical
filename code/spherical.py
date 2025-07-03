"""
Implements the forward modeling and equivalent sources

Also includes some utilities for general use.
"""

import numpy as np
import numba
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


class EquivalentSourcesMagGeod:

    def __init__(
        self,
        damping=None,
        depth=0,
        inclination=90,
        declination=0,
        ellipsoid=bl.WGS84,
        source_coordinates=None,
    ):
        self.damping = damping
        self.depth = depth
        self.inclination = inclination
        self.declination = declination
        self.ellipsoid = ellipsoid
        self.source_coordinates = source_coordinates

    def fit(self, coordinates, inclination, declination, data, weights=None):
        """ """
        coordinates = tuple(np.atleast_1d(c).ravel() for c in coordinates)
        data = data.ravel()
        jacobian = self.jacobian(coordinates, inclination, declination)
        moment_amplitude = verde.base.least_squares(
            jacobian, data, weights=weights, damping=self.damping
        )
        self.dipole_moments_ = hm.magnetic_angles_to_vec(
            moment_amplitude, self.inclination, self.declination
        )
        return self

    def jacobian(self, coordinates, inclination, declination):
        """ """
        if self.source_coordinates is None:
            self.source_coordinates_ = (
                coordinates[0].copy(),
                coordinates[1].copy(),
                coordinates[2] - self.depth,
            )
        else:
            self.source_coordinates_ = self.source_coordinates
        n_data = coordinates[0].size
        n_params = self.source_coordinates_[0].size
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

    def predict(self, coordinates):
        if not hasattr(self, "dipole_moments_"):
            raise ValueError("Fit the class before predicting.")
        result = dipole_magnetic_geodetic(
            coordinates, self.source_coordinates_, self.dipole_moments_
        )
        return result


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
