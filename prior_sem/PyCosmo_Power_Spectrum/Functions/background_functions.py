import pidly
import numpy as np

# Definition of a directory where we save the results from iCosmo
output_directory = 'output/'

def run_icosmo_background(cosmo_instance, z_max):
    
    """
    This function computes the comoving radius with iCosmo. 
    :param cosmo_instance: instance of PyCosmo, used to setup the cosmology;
    :param z_max: maximum redshift valued for the computations;
    :return chi(a): comoving radius in units of [Mpc]
    """
    
    # icosmo instance with GDL
    idl = pidly.IDL('/usr/bin/gdl', idl_prompt='GDL> ')
    
    # Cosmological setup
    idl('fid = set_fiducial(cosmo_in={omega_m:'+ str(cosmo_instance.params.omega_m) 
        + 'd,omega_b:' + str(cosmo_instance.params.omega_b) 
        + 'd,omega_l:' + str(1-cosmo_instance.params.omega_m) + 'd,h:'+ str(cosmo_instance.params.h) 
        + 'd,w0:-1.0d,sigma8:0.8d,n:1.d}, calc_in={fit_nl:'+ str(2) + ',fit_tk:'+ str(1)
        + ',nz_fn:5000,ran_z:[0.0d,' + str(z_max) + 'd],speed:0})')
    
    # Calculate the cosmological obervables
    idl('cosmo=mk_cosmo(fid)')

    # Save the output for the comoving radius (CHI)
    nameout = output_directory + 'chi_icosmo'
    param = 'CHI'
    idl("openw,lun,'" + nameout + ".txt',/get_lun")
    idl("printf,lun,cosmo.evol." + param + ",format='(d,x)'")
    idl('Free_lun,lun')
    
    # Save the Hubble Radius (R0) to be multiplied to the comoving distance
    nameout_R0 = output_directory + "R0_icosmo"
    idl("openw,lun,'" + nameout_R0 + ".txt',/get_lun")
    idl("printf,lun,cosmo.const.R0,format='(d,x)'")
    idl('Free_lun,lun')
    
    idl.close()

    # Read the results
    R0_icosmo = np.loadtxt(nameout_R0 + '.txt')
    chi_icosmo_temp  = np.loadtxt(nameout + '.txt')
    chi_icosmo_output = chi_icosmo_temp*R0_icosmo
    
    return chi_icosmo_output
    