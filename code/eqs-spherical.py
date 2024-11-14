def foward_modeling_spherical(coordinates,
                                          dipoles,
                                          Inclination,
                                          Declination,
                                          magnetic_moment):
    """
    Computes the magnetic field components in spherical coordinates at observation points 
    due to a set of dipoles, each with a specified magnetic moment.

    Parameters:
    - coordinates (tuple of arrays): A tuple containing the spherical coordinates (longitude, latitude, radius) 
      of the observation points. Each coordinate should be a 1D array.
    - dipoles (tuple of arrays): A tuple containing the spherical coordinates (longitude, latitude, radius) 
      of the source dipoles. Each dipole's position is described by a 1D array for longitude, latitude, and radius.
    - magnetic_moments (tuple of arrays): A tuple containing the magnetic moments of the dipoles, given by three 
      components. Each component is a 1D array.

    Returns:
    - B_r (array): Magnetic field component in the radial direction (nT) at each observation point.
    - B_theta (array): Magnetic field component in the theta direction (nT) at each observation point.
    - B_phi (array): Magnetic field component in the phi direction (nT) at each observation point.

    This function:
    1. Converts latitude and longitude from degrees to radians for spherical coordinate calculations.
    2. Calculates directional cosines and separation distances between source and observation points.
    3. Computes intermediate matrices (H_ij) accounting for the geometry and separation between source and observation points.
    4. Uses the magnetic moments and geometry to compute the magnetic field components in spherical coordinates.

    Notes:
    - The magnetic field components are returned in nanotesla (nT).
    - The constant CM (not passed as an argument) is used to scale the magnetic field based on the source-observation geometry.

    The function assumes that the magnetic field is modeled by the superposition of dipoles with specified magnetic moments. 
    Each dipole is characterized by its position in spherical coordinates (longitude, latitude, radius) and its magnetic moment components.
    """
    longitude_observed_data, latitude_observed_data, radius_observed_data = tuple(c.ravel() for c in coordinates)
    longitude_source, latitude_source, radius_source = [np.asarray(c).ravel() for c in dipoles]#tuple(c.ravel() for c in dipoles)
    
    m_jr = - magnetic_moment * np.sin(Inclination)
    m_jtheta = - magnetic_moment * np.cos(Inclination) * np.cos(Declination)
    m_jphi = magnetic_moment * np.cos(Inclination) * np.cos(Declination)
    
    colatitude_obs = np.pi/2 - np.deg2rad(latitude_observed_data)
    longitude_obs = np.deg2rad(longitude_observed_data)
    colatitude_source = np.pi/2 - np.deg2rad(latitude_source)
    longitude_source = np.deg2rad(longitude_source)

    # Calculate the directional cossines
    diff_long_obs_long_source = longitude_obs - longitude_source
    mu_ij = np.cos(colatitude_obs) * np.cos(colatitude_source) + np.sin(colatitude_obs) * np.sin(colatitude_source) * np.cos(diff_long_obs_long_source)
    ri_dot_thetaj = -np.cos(colatitude_obs) * np.sin(colatitude_source) + np.sin(colatitude_obs) * np.cos(colatitude_source) * np.cos(diff_long_obs_long_source)
    ri_dot_phij = np.sin(colatitude_obs) * np.sin(diff_long_obs_long_source)
    thetai_dot_rj = -np.sin(colatitude_obs) * np.cos(colatitude_source) + np.cos(colatitude_obs) * np.sin(colatitude_source) * np.cos(diff_long_obs_long_source)
    thetai_dot_thetaj = np.sin(colatitude_obs) * np.sin(colatitude_source) + np.cos(colatitude_obs) * np.cos(colatitude_source) * np.cos(diff_long_obs_long_source)
    thetai_dot_phij = np.cos(colatitude_obs) * np.sin(diff_long_obs_long_source)
    phii_dot_rj = -np.sin(colatitude_source) * np.sin(diff_long_obs_long_source)
    phii_dot_thetaj = -np.cos(colatitude_source) * np.sin(diff_long_obs_long_source)
    phii_dot_phij = np.cos(diff_long_obs_long_source)
    
    # Distance r_ij between points
    r_ij = np.sqrt((radius_observed_data ** 2) + (radius_source ** 2) - 2 * radius_observed_data * radius_source * mu_ij)
    
    # Define magnetic field terms
    CONSTANT = CM / r_ij ** 3
    H_11 = CONSTANT * (3 * ((radius_observed_data - radius_source * mu_ij) * (radius_observed_data * mu_ij - radius_source) / r_ij ** 2) - mu_ij)
    H_12 = CONSTANT * (3 * ((radius_observed_data - radius_source * mu_ij) * (radius_observed_data * ri_dot_thetaj) / r_ij ** 2) - ri_dot_thetaj)
    H_13 = CONSTANT * (3 * ((radius_observed_data - radius_source * mu_ij) * (radius_observed_data * ri_dot_phij) / r_ij ** 2) - ri_dot_phij)
    H_21 = -CONSTANT * (3 * ((radius_source * thetai_dot_rj) * (radius_observed_data * mu_ij - radius_source) / r_ij ** 2) + thetai_dot_rj)
    H_22 = -CONSTANT * (3 * ((radius_source * thetai_dot_rj) * (radius_observed_data * ri_dot_thetaj) / r_ij ** 2) + thetai_dot_thetaj)
    H_23 = -CONSTANT * (3 * ((radius_source * thetai_dot_rj) * (radius_observed_data * ri_dot_phij) / r_ij ** 2) + thetai_dot_phij)
    H_31 = -CONSTANT * (3 * ((radius_source * phii_dot_rj) * (radius_observed_data * mu_ij - radius_source) / r_ij ** 2) + phii_dot_rj)
    H_32 = -CONSTANT * (3 * ((radius_source * phii_dot_rj) * (radius_observed_data * ri_dot_thetaj) / r_ij ** 2) + phii_dot_thetaj)
    H_33 = -CONSTANT * (3 * ((radius_source * phii_dot_rj * radius_observed_data * ri_dot_phij) / r_ij ** 2) + phii_dot_phij)
    
    # Calculate magnetic components for each field direction
    mj_ri = m_jr * mu_ij + m_jtheta * ri_dot_thetaj + m_jphi * ri_dot_phij
    mj_thetai = m_jr * thetai_dot_rj + m_jtheta * thetai_dot_thetaj + m_jphi * thetai_dot_phij
    mj_phii = m_jr * phii_dot_rj + m_jtheta * phii_dot_thetaj + m_jphi * phii_dot_phij

    
    # Calculate final magnetic field components
    B_r = (H_11 * mj_ri + H_12 * mj_thetai + H_13 * mj_phii) * 1e9
    B_theta = (H_21 * mj_ri + H_22 * mj_thetai + H_23 * mj_phii) * 1e9
    B_phi = (H_31 * mj_ri + H_32 * mj_thetai + H_33 * mj_phii) * 1e9

    return B_r, B_theta, B_phi