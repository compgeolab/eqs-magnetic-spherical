"""
Implements the forward modeling and equivalent sources

Also includes some utilities for general use.
"""

import numpy as np
import harmonica as hm
import choclo
import bordado as bd
import numba
import boule as bl


CM = choclo.constants.VACUUM_MAGNETIC_PERMEABILITY / 4 / np.pi


def vector_geodetic_to_spherical(
    latitude_spherical, latitude, eastward, northward, upward
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
    latitude_spherical, latitude, eastward, northward, radial
):
    """
    Rotate a vector from a spherical to a geodetic system.
    """
    return vector_geodetic_to_spherical(
        latitude, latitude_spherical, eastward, northward, radial
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
        latitude_spherical=dipoles_sph[1],
        latitude=dipoles[1],
        eastward=magnetic_moments[0],
        northward=magnetic_moments[1],
        upward=magnetic_moments[2],
    )
    b_lon, b_lat, b_radial = dipole_magnetic_spherical(
        coordinates_sph, dipoles_sph, magnetic_moments_sph
    )
    b_lon, b_lat, b_height = vector_spherical_to_geodetic(
        latitude_spherical=coordinates_sph[1],
        latitude=coordinates[1],
        eastward=b_lon,
        northward=b_lat,
        radial=b_radial,
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


def jacobian(
    coordinates,
    dipoles,
    inclination_source,
    declination_souce,
    inclination_field,
    declination_field,
):
    coordinates = bd.check_coordinates(coordinates)
    dipoles = bd.check_coordinates(dipoles)
    coordinates = tuple(c.ravel() for c in coordinates)
    dipoles = tuple(c.ravel() for c in dipoles)
    n_data = coordinates[0].size
    n_params = dipoles[0].size

    unit_magnetic_moment = hm.magnetic_angles_to_vec(
        1, inclination_source, declination_souce
    )
    main_field = hm.magnetic_angles_to_vec(1, inclination_field, declination_field)

    A = np.empty((n_data, n_params))
    _jacobian_fast(
        np.radians(coordinates[0]),
        np.radians(90 - coordinates[1]),
        coordinates[2],
        np.radians(dipoles[0]),
        np.radians(90 - dipoles[1]),
        dipoles[2],
        unit_magnetic_moment[0],
        unit_magnetic_moment[1],
        unit_magnetic_moment[2],
        main_field[0],
        main_field[1],
        main_field[2],
        A,
    )

    return A


@numba.jit(nopython=True, parallel=True)
def _jacobian_fast(
    longitude,
    colatitude,
    radius,
    longitude_d,
    colatitude_d,
    radius_d,
    m_lon,
    m_colat,
    m_radial,
    f_lon,
    f_lat,
    f_radial,
    A,
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
            b_lon, b_lat, b_radial = _kernel(
                cos_lon,
                sin_lon,
                cos_colat,
                sin_colat,
                radius[i],
                cos_colat_d,
                sin_colat_d,
                radius_d[j],
                m_lon,
                m_colat,
                m_radial,
            )
            A[i, j] = f_lon * b_lon + f_lat * b_lat + f_radial * b_radial


def calculate_coefficients(observed_data, A, damping):

    I = np.identity(A.shape[1])  # needs to = m x m
    # np.shape(A) = A.shape

    # The @ operator can be used for cocalculate_total_field_anomaly(grid_coord, observed_source_coord, 0, 0, 1e17, 0, 0)nventional matrix multiplication.
    system_matrix = A.T @ A + I * damping
    system_rhs_vector = A.T @ observed_data

    coefficients = np.linalg.solve(system_matrix, system_rhs_vector)

    return coefficients


def profile_points(start, end, npoints, depth=0):
    """
    Generate evenly spaced coordinates for a profile along a great circle,
    with a fixed depth value for the entire arc.

    Both start and end should be (longitude, latitude) pairs.
    The depth parameter sets a constant depth for all points along the profile.

    Returns longitude, latitude, and depth coordinates in a format that can be
    passed to xarray.Dataset.interp.
    """
    lon1, lat1 = np.radians(start)
    lon2, lat2 = np.radians(end)

    azimuth1 = np.arctan2(
        np.cos(lat2) * np.sin(lon2 - lon1),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1),
    )

    azimuth_equator = np.arctan2(
        np.sin(azimuth1) * np.cos(lat1),
        np.sqrt(np.cos(azimuth1) ** 2 + np.sin(azimuth1) ** 2 * np.sin(lat1) ** 2),
    )

    great_circle_equator = np.arctan2(np.tan(lat1), np.cos(azimuth1))
    lon_equator = lon1 - np.arctan2(
        np.sin(azimuth_equator) * np.sin(great_circle_equator),
        np.cos(great_circle_equator),
    )

    great_circle_distance = 2 * np.arcsin(
        np.sqrt(
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
    )

    distances = np.linspace(start=0, stop=great_circle_distance, num=npoints)
    distances_equator = distances + great_circle_equator

    latitudes = np.arctan2(
        np.cos(azimuth_equator) * np.sin(distances_equator),
        np.sqrt(
            np.cos(distances_equator) ** 2
            + (np.sin(azimuth_equator) * np.sin(distances_equator)) ** 2
        ),
    )

    longitudes = lon_equator + np.arctan2(
        np.sin(azimuth_equator) * np.sin(distances_equator), np.cos(distances_equator)
    )

    longitude = np.degrees(longitudes)
    latitude = np.degrees(latitudes)
    depth_array = np.full(npoints, depth)

    dike = np.array([longitude]), np.array([latitude]), np.array([depth_array])

    return dike
