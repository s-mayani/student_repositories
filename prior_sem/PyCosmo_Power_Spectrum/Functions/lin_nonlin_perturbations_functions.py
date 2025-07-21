import pidly
import numpy as np

# Definition of a directory where we save the results from iCosmo
output_directory = 'output/'

def run_icosmo_growth_factor(cosmo_instance, z_max):
    
    """
    This function computes the growth factor with iCosmo. 
    :param cosmo_instance: instance of PyCosmo, used to setup the cosmology;
    :return D(a): growth factor normalised to 1 at a=1.
    """
        
    # icosmo instance with GDL
    idl = pidly.IDL('/usr/bin/gdl', idl_prompt='GDL> ')
    
    # Cosmology setup
    idl('fid = set_fiducial(cosmo_in={omega_m:'+ str(cosmo_instance.params.omega_m) 
            + 'd,omega_b:' + str(cosmo_instance.params.omega_b) 
            + 'd,omega_l:' + str(1-cosmo_instance.params.omega_m) + 'd,h:'+ str(cosmo_instance.params.h) 
            + 'd,w0:-1.0d,sigma8:0.8d,n:1.d}, calc_in={fit_nl:'+ str(2) + ',fit_tk:'+ str(1)
            + ',nz_fn:5000,ran_z:[0.0d,' + str(z_max) + 'd],speed:0})')
        
    # Calculate the cosmological quantities
    idl('cosmo=mk_cosmo(fid)')

    # Save the growth factor array (D)
    nameout = output_directory + 'D_a_icosmo'
    param = 'D'
    idl("openw,lun,'" + nameout + ".txt',/get_lun")
    idl("printf,lun,cosmo.evol." + param + ",format='(d,x)'")
    idl('Free_lun,lun')
    
    # Save the scale factor array (A)
    nameout_a = output_directory + "a_icosmo"
    idl("openw,lun,'" + nameout_a + ".txt',/get_lun")
    idl("printf,lun,cosmo.evol.A,format='(d,x)'")
    idl('Free_lun,lun')
        
    # Read the results
    a_icosmo = np.loadtxt(nameout_a + '.txt')
    D_icosmo  = np.loadtxt(nameout + '.txt')
        
    return a_icosmo, D_icosmo


def run_icosmo_pk(cosmo_instance, model_lin='EH', model_nonlin='Smith'):
    
    """
    This function computes the linear and non-linear power spectrum with iCosmo, at redshift z=0,1,2.
    :param cosmo_instance: instance of PyCosmo, used to setup the cosmology;
    :param model_lin: sets the linear fitting function. It can be either "EH" or "BBKS";
    :param model_nonlin: sets the non-linear fitting function. It can be either "PD", "MA", "Smith",
    standing or "Peacock & Dodds", "Ma et al." and "Smith et al.", respectively;
    :return pk_lin_output, pk_nonlin_output, ks_icosmo: collections of linear and non-linear power spectra
    at z=0,1,2, in units of [Mpc]^{3}, and wavenumbers in units of [Mpc]^{-1}.
    """
    
    # Setup of the linear fitting function
    if model_lin=='EH':
        fit_tk_lin = 1
    elif model_lin=='BBKS':
        fit_tk_lin = 2
    else:
        raise ValueError("No available fitting functions for the input linear Tk")
        
    # Setup of the linear fitting function
    if model_nonlin=='PD':
        fit_tk_nonlin = 0
    elif model_nonlin=='MA':
        fit_tk_nonlin = 1
    elif model_nonlin=='Smith':
        fit_tk_nonlin = 2
    else:
        raise ValueError("No available fitting functions for the input non-linear Tk")
        
    # GDL setup
    idl = pidly.IDL('/usr/bin/gdl', idl_prompt='GDL> ')
    
    # Cosmological setup
    idl('fid = set_fiducial(cosmo_in={omega_m:'+ str(cosmo_instance.params.omega_m) 
        + 'd,omega_b:' + str(cosmo_instance.params.omega_b) 
        + 'd,omega_l:' + str(1-cosmo_instance.params.omega_m) + 'd,h:'+ str(cosmo_instance.params.h) 
        + 'd,w0:-1.0d,sigma8:0.8d,n:1.d}, calc_in={fit_nl:'+ str(fit_tk_nonlin) + ',fit_tk:'+ str(fit_tk_lin)
        + '})') 
    
    # Calculate the cosmological quantities
    idl('cosmo=mk_cosmo(fid)')
    
    # Save the wavenumbers
    nameout_ks = output_directory + 'ks_icosmo'
    idl("openw,lun,'" + nameout_ks + "',/get_lun")
    idl("printf,lun,cosmo.pk.k,format='(d,x)'")
    idl('Free_lun,lun')
    
    
    # Collect the linear and the non-linear P(k) at redshift z=0
    nameout_pk_lin_0 = output_directory + 'lin_pk_icosmo_z0'
    idl("openw,lun,'" + nameout_pk_lin_0 + "',/get_lun")
    idl("printf,lun,cosmo.pk.pk_l[*,0],format='(d,x)'")
    idl('Free_lun,lun')
    
    nameout_pk_nonlin_0 = output_directory + 'nonlin_pk_icosmo_z0'
    idl("openw,lun,'" + nameout_pk_nonlin_0 + "',/get_lun")
    idl("printf,lun,cosmo.pk.pk[*,0],format='(d,x)'")
    idl('Free_lun,lun')
    
    
    # Collect the linear and the non-linear P(k) at redshift z=1
    nameout_pk_lin_1 = output_directory + 'lin_pk_icosmo_z1'
    idl("openw,lun,'" + nameout_pk_lin_1 + "',/get_lun")
    idl("printf,lun,cosmo.pk.pk_l[*,20],format='(d,x)'")
    idl('Free_lun,lun')
    
    nameout_pk_nonlin_1 = output_directory + 'nonlin_pk_icosmo_z1'
    idl("openw,lun,'" + nameout_pk_nonlin_1 + "',/get_lun")
    idl("printf,lun,cosmo.pk.pk[*,20],format='(d,x)'")
    idl('Free_lun,lun')
    
    
    # Collect the linear and the non-linear P(k) at redshift z=1
    nameout_pk_lin_2 = output_directory + 'lin_pk_icosmo_z2'
    idl("openw,lun,'" + nameout_pk_lin_2 + "',/get_lun")
    idl("printf,lun,cosmo.pk.pk_l[*,40],format='(d,x)'")
    idl('Free_lun,lun')
    
    nameout_pk_nonlin_2 = output_directory + 'nonlin_pk_icosmo_z2'
    idl("openw,lun,'" + nameout_pk_nonlin_2 + "',/get_lun")
    idl("printf,lun,cosmo.pk.pk[*,40],format='(d,x)'")
    idl('Free_lun,lun')


    # Read and collect the results
    lin_pk_z0 = np.loadtxt(nameout_pk_lin_0)*1./(cosmo_instance.params.h**3)
    lin_pk_z1 = np.loadtxt(nameout_pk_lin_1)*1./(cosmo_instance.params.h**3)
    lin_pk_z2 = np.loadtxt(nameout_pk_lin_2)*1./(cosmo_instance.params.h**3)
    nonlin_pk_z0 = np.loadtxt(nameout_pk_nonlin_0)*1./(cosmo_instance.params.h**3)
    nonlin_pk_z1 = np.loadtxt(nameout_pk_nonlin_1)*1./(cosmo_instance.params.h**3)
    nonlin_pk_z2 = np.loadtxt(nameout_pk_nonlin_2)*1./(cosmo_instance.params.h**3)
    ks_icosmo  = np.loadtxt(nameout_ks)*cosmo_instance.params.h
    
    pk_lin_output = np.array([lin_pk_z0, lin_pk_z1, lin_pk_z2])
    pk_nonlin_output = np.array([nonlin_pk_z0, nonlin_pk_z1, nonlin_pk_z2])

    #idl.close()
    
    return pk_lin_output, pk_nonlin_output, ks_icosmo
    
    