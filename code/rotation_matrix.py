'''Switched up code for Jet (thx Shriya for finding it) to AAU with the correct roation matrix. All can be found here https://github.com/norashipp/stream_search/blob/master/code/stream_data.py 
Also added some stuff from Noras repo.'''

import jax
import jax.numpy as jnp
import numpy as np

def icrs_to_stream(ra_rad, dec_rad, rotmat=[[0.83697865, 0.29481904, -0.4610298], [0.51616778, -0.70514011, 0.4861566], [0.18176238, 0.64487142, 0.74236331]]):
    """
    define a *differentiable* coordinate transfrom from ra and dec --> stream phi1, phi2 (defaults AAU)
    ra_rad: icrs ra [radians]
    dec_red: icrs dec [radians]
    """
    R = jnp.array(rotmat)

    icrs_vec = jnp.vstack(
        [
            jnp.cos(ra_rad) * jnp.cos(dec_rad),
            jnp.sin(ra_rad) * jnp.cos(dec_rad),
            jnp.sin(dec_rad),
        ]
    ).T

    stream_frame_vec = jnp.einsum("ij,kj->ki", R, icrs_vec)

    phi1 = jnp.arctan2(stream_frame_vec[:, 1], stream_frame_vec[:, 0]) * 180 / jnp.pi
    phi2 = jnp.arcsin(stream_frame_vec[:, 2]) * 180 / jnp.pi

    return phi1, phi2


@jax.jit
def stream_to_icrs(phi1, phi2, rotmat=[[0.83697865, 0.29481904, -0.4610298], [0.51616778, -0.70514011, 0.4861566], [0.18176238, 0.64487142, 0.74236331]]):
    """
    define a *differentiable* coordinate transform from stream (AAU default) phi1, phi2 --> ra and dec
    Using the inverse rotation matrix
    phi1: aau phi1 [degrees]
    phi2: aau phi2 [degrees]
    """
    R = jnp.array(rotmat)

    # Convert phi1, phi2 to radians
    phi1_rad = phi1 * jnp.pi / 180
    phi2_rad = phi2 * jnp.pi / 180

    # Stream frame vector
    stream_frame_vec = jnp.vstack(
        [
            jnp.cos(phi2_rad) * jnp.cos(phi1_rad),
            jnp.cos(phi2_rad) * jnp.sin(phi1_rad),
            jnp.sin(phi2_rad),
        ]
    ).T

    # Transform back to ICRS frame using the inverse of R
    icrs_vec = jnp.einsum("ij,kj->ki", R.T, stream_frame_vec)

    # Compute ra and dec in radians
    ra_rad = jnp.arctan2(icrs_vec[:, 1], icrs_vec[:, 0])
    dec_rad = jnp.arcsin(icrs_vec[:, 2])

    return ra_rad, dec_rad

def phi12_rotmat(alpha,delta,R_phi12_radec):
    '''
    Nora's Code. It's better. Converts coordinates (alpha,delta) to ones defined by a rotation matrix R_phi12_radec, applied on the original coordinates

    Critical: All angles must be in degrees
    '''
    
    vec_radec = np.array([np.cos(alpha*np.pi/180.)*np.cos(delta*np.pi/180.),np.sin(alpha*np.pi/180.)*np.cos(delta*np.pi/180.),np.sin(delta*np.pi/180.)])

    vec_phi12 = np.zeros(np.shape(vec_radec))
    
    vec_phi12[0] = np.sum(R_phi12_radec[0][i]*vec_radec[i] for i in range(3))
    vec_phi12[1] = np.sum(R_phi12_radec[1][i]*vec_radec[i] for i in range(3))
    vec_phi12[2] = np.sum(R_phi12_radec[2][i]*vec_radec[i] for i in range(3))
    
    vec_phi12 = vec_phi12.T

    vec_phi12 = np.dot(R_phi12_radec,vec_radec).T

    phi1 = np.arctan2(vec_phi12[:,1],vec_phi12[:,0])*180./np.pi
    phi2 = np.arcsin(vec_phi12[:,2])*180./np.pi


    return [phi1,phi2]

def pmphi12(alpha,delta,mu_alpha_cos_delta,mu_delta,R_phi12_radec):
    '''
    Converts proper motions (mu_alpha_cos_delta,mu_delta) to those in coordinates defined by the rotation matrix, R_phi12_radec, applied to the original coordinates

    Critical: All angles must be in degrees
    '''
    
    k_mu = 4.74047

    phi1,phi2 = phi12_rotmat(alpha,delta,R_phi12_radec)


    r = np.ones(len(alpha))

    vec_v_radec = np.array([np.zeros(len(alpha)),k_mu*mu_alpha_cos_delta*r,k_mu*mu_delta*r]).T

    worker = np.zeros((len(alpha),3))

    worker[:,0] = ( np.cos(alpha*np.pi/180.)*np.cos(delta*np.pi/180.)*vec_v_radec[:,0]
                   -np.sin(alpha*np.pi/180.)*vec_v_radec[:,1]
                   -np.cos(alpha*np.pi/180.)*np.sin(delta*np.pi/180.)*vec_v_radec[:,2] )

    worker[:,1] = ( np.sin(alpha*np.pi/180.)*np.cos(delta*np.pi/180.)*vec_v_radec[:,0]
                   +np.cos(alpha*np.pi/180.)*vec_v_radec[:,1]
                   -np.sin(alpha*np.pi/180.)*np.sin(delta*np.pi/180.)*vec_v_radec[:,2] )

    worker[:,2] = ( np.sin(delta*np.pi/180.)*vec_v_radec[:,0]
                   +np.cos(delta*np.pi/180.)*vec_v_radec[:,2] )

    worker2 = np.zeros((len(alpha),3))

    worker2[:,0] = np.sum(R_phi12_radec[0][axis]*worker[:,axis] for axis in range(3))
    worker2[:,1] = np.sum(R_phi12_radec[1][axis]*worker[:,axis] for axis in range(3))
    worker2[:,2] = np.sum(R_phi12_radec[2][axis]*worker[:,axis] for axis in range(3))

    worker[:,0] = ( np.cos(phi1*np.pi/180.)*np.cos(phi2*np.pi/180.)*worker2[:,0]
                   +np.sin(phi1*np.pi/180.)*np.cos(phi2*np.pi/180.)*worker2[:,1]
                   +np.sin(phi2*np.pi/180.)*worker2[:,2] )

    worker[:,1] = (-np.sin(phi1*np.pi/180.)*worker2[:,0]
                   +np.cos(phi1*np.pi/180.)*worker2[:,1] )
                   

    worker[:,2] = (-np.cos(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*worker2[:,0]
                   -np.sin(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*worker2[:,1]
                   +np.cos(phi2*np.pi/180.)*worker2[:,2] )

    mu_phi1_cos_delta = worker[:,1]/(k_mu*r)
    mu_phi2 = worker[:,2]/(k_mu*r)

    return mu_phi1_cos_delta, mu_phi2

def pmphi12_reflex(alpha,delta,mu_alpha_cos_delta,mu_delta,R_phi12_radec,dist,vlsr=np.array([11.1, 240., 7.3])):
    
    ''' 
    returns proper motions in coordinates defined by R_phi12_radec transformation corrected by the Sun's reflex motion
    all angles must be in degrees
     vlsr = np.array([11.1,240.,7.3]) 
    '''

    k_mu = 4.74047

    a_g = np.array([[-0.0548755604, +0.4941094279, -0.8676661490],
                    [-0.8734370902, -0.4448296300, -0.1980763734], 
                    [-0.4838350155, 0.7469822445, +0.4559837762]])

    nvlsr = -vlsr

    phi1, phi2 = phi12_rotmat(alpha,delta,R_phi12_radec)

    phi1 = phi1*np.pi/180.
    phi2 = phi2*np.pi/180.

    pmphi1, pmphi2 = pmphi12(alpha,delta,mu_alpha_cos_delta,mu_delta,R_phi12_radec)

    M_UVW_phi12 = np.array([[np.cos(phi1)*np.cos(phi2),-np.sin(phi1),-np.cos(phi1)*np.sin(phi2)],
                            [np.sin(phi1)*np.cos(phi2), np.cos(phi1),-np.sin(phi1)*np.sin(phi2)],
                            [     np.sin(phi2)        ,     np.zeros_like(phi1)     , np.cos(phi2)]])

    vec_nvlsr_phi12 = np.dot(M_UVW_phi12.T,np.dot(R_phi12_radec,np.dot(a_g,nvlsr)))

    # return pmphi1 - vec_nvlsr_phi12[1]/(k_mu*dist), pmphi2 - vec_nvlsr_phi12[2]/(k_mu*dist)
    return pmphi1 - vec_nvlsr_phi12[:,1]/(k_mu*dist), pmphi2 - vec_nvlsr_phi12[:,2]/(k_mu*dist)