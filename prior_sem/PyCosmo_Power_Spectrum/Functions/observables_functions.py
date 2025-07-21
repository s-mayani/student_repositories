import pidly
import numpy as np
import pyccl as ccl

# Definition of a directory where we save the results from iCosmo
output_directory = 'output/'

def run_icosmo_cls(cosmo_instance, pklin_type='EH'):
    
    """
    This function computes angular power spectrum Cls gamma gamma.
    :param cosmo_instance: instance of PyCosmo, used to setup the cosmology;
    :param pklin_type: sets the linear fitting function. It can be either 1 (EH) or 2 (BBKS);
    :return ells, c_ells_icosmo: ells and Cls. 
    """
        
    output_directory = 'output/'
        
    # icosmo instance with GDL
    idl = pidly.IDL('/usr/bin/gdl', idl_prompt='GDL> ')
                   
    # Setup the linear transfer function
    if pklin_type=='EH':
        lin_tk = 1
    elif pklin_type=='BBKS':
        lin_tk = 2
    else:
        raise ValueError("No available fitting functions for the input linear Tk")
        
    # Cosmology setup
    idl('fid = set_fiducial(cosmo_in={omega_m:'+ str(cosmo_instance.params.omega_m) 
        + 'd,omega_b:' + str(cosmo_instance.params.omega_b) 
        + 'd,omega_l:' + str(1-cosmo_instance.params.omega_m) + 'd,h:'+ str(cosmo_instance.params.h) 
        + 'd,w0:-1.0d,sigma8:0.8d,n:1.d}, calc_in={fit_nl:'+ str(2) + ',fit_tk:'+ str(lin_tk)
        + ',nz_fn:5000,ran_z:[0.0d,5.0d],speed:0},'
        + 'expt_in={sv1_n_zbin:1,sv1_zerror:0.0d,sv1_z_med:1.23,sv1_dndzp:[2.d,2.d]})')
         
    # Calculate the cosmological quantities
    idl('sv=mk_survey(fid,"sv1")')
    idl('cosmo=mk_cosmo(fid)')
    idl('cl=mk_cl_tomo(fid,cosmo,sv)')

    # Save the ells array
    nameout = output_directory + 'ells_icosmo'
    param = 'L'
    idl("openw,lun,'" + nameout + ".txt',/get_lun")
    idl("printf,lun,cl." + param + ",format='(d,x)'")
    idl('Free_lun,lun')

    # Save the Cls array
    nameout_a = output_directory + "c_ells_icosmo"
    idl("openw,lun,'" + nameout_a + ".txt',/get_lun")
    idl("printf,lun,cl.CL,format='(d,x)'")
    idl('Free_lun,lun')
        
    # Read the results
    c_ells_icosmo = np.loadtxt(nameout_a + '.txt')
    ells  = np.loadtxt(nameout + '.txt')
        
    return ells, c_ells_icosmo


def ccl_cls(cosmo_instance, linear_tk, pk, ells):
    
    """
        This function computes angular power spectrum Cls gamma gamma with CCL, using the redshift distributions
        from the CCL GitHub Benchmark Comparisons: 
        https://github.com/LSSTDESC/CCL/blob/master/examples/Benchmark_Comparisons.ipynb.
        :param cosmo_instance: instance of PyCosmo, used to setup the cosmology;
        :param linear_tk: sets the linear fitting function. It can be either 'eisenstein_hu' or 'bbks';
        :param pk: sets the non-linear fitting function. It can be either 'halofit', for example;
        :return c_ells_icosmo: Cls computed with CCL. 
    """
    
    # CCL cosmology setup
    cosmo_cls = ccl.Cosmology(h = cosmo_instance.params.h,
                          Omega_b = cosmo_instance.params.omega_b,
                          Omega_c = cosmo_instance.params.omega_m - cosmo_instance.params.omega_b,
                          w0 = cosmo_instance.params.w0,
                          wa = cosmo_instance.params.wa,
                          Neff = cosmo_instance.params.Nnu,
                          n_s = cosmo_instance.params.n,
                          sigma8 = cosmo_instance.params.pk_norm,
                          transfer_function = linear_tk, # 'eisenstein_hu'
                          matter_power_spectrum = pk,
                          mass_function = 'shethtormen',
                          baryons_power_spectrum='nobaryons')
    
    accuracy = 1e-9

    # ----------------------
    #Redshift distribution
    zmean1=1.0; zmean2=1.5;
    sigz1=0.15; sigz2=0.15;
    nzs=512;

    # Analytic redshift distributions
    z_a_1=np.linspace(zmean1-5*sigz1,zmean1+5*sigz1,nzs);
    z_a_2=np.linspace(zmean2-5*sigz2,zmean2+5*sigz2,nzs);
    pz_a_1=np.exp(-0.5*((z_a_1-zmean1)/sigz1)**2)
    pz_a_2=np.exp(-0.5*((z_a_2-zmean2)/sigz2)**2)

    # Bias parameters for these distributions
    bz_a_1=np.ones_like(z_a_1); bz_a_2=np.ones_like(z_a_2);

    # Binned redshift distributions and biases
    z_h_1,pz_h_1=np.loadtxt("Cls_nz_bins/bin1_histo.txt",unpack=True)[:,1:]
    z_h_2,pz_h_2=np.loadtxt("Cls_nz_bins/bin2_histo.txt",unpack=True)[:,1:]
    bz_h_1=np.ones_like(z_h_1); bz_h_2=np.ones_like(z_h_2);
    # ----------------------


    zarrs={'analytic':{'b1':{'z':z_a_1,'nz':pz_a_1,'bz':bz_a_1},'b2':{'z':z_a_2,'nz':pz_a_2,'bz':bz_a_2}},
          'histo':{'b1':{'z':z_h_1,'nz':pz_h_1,'bz':bz_h_1},'b2':{'z':z_h_2,'nz':pz_h_2,'bz':bz_h_2}}}

    # Initialize tracers
    trcrs={}
    for nztyp in ['analytic','histo'] :
        trcrs[nztyp]={}
        za=zarrs[nztyp]
        trcrs[nztyp]['nc_1']=ccl.NumberCountsTracer(cosmo_cls,has_rsd=False,
                                                    dndz=(za['b1']['z'],za['b1']['nz']),
                                                    bias=(za['b1']['z'],za['b1']['bz']))
        trcrs[nztyp]['nc_2']=ccl.NumberCountsTracer(cosmo_cls,has_rsd=False,
                                                    dndz=(za['b2']['z'],za['b2']['nz']),
                                                    bias=(za['b2']['z'],za['b2']['bz']))
        trcrs[nztyp]['wl_1']=ccl.WeakLensingTracer(cosmo_cls,
                                                   dndz=(za['b1']['z'],za['b1']['nz']))
        trcrs[nztyp]['wl_2']=ccl.WeakLensingTracer(cosmo_cls,
                                                   dndz=(za['b2']['z'],za['b2']['nz']))
        


    cl_ccl={}
    el_ccl={}
    # ells=np.loadtxt("larr_cls.txt").astype(int)
    for nztyp in ['analytic','histo'] :
        #Limber prefactors
        lf_dl=(ells+0.5)**2/np.sqrt((ells+2.)*(ells+1.)*ells*(ells-1.))
        lf_dc=(ells+0.5)**2/(ells*(ells+1.))
        lf_ll=ells*(ells+1.)/np.sqrt((ells+2.)*(ells+1.)*ells*(ells-1.))

        #Power spectra
        cl_ccl[nztyp]={}
        cl_ccl[nztyp]['wl_1/wl_1']=ccl.angular_cl(cosmo_cls,
                                           trcrs[nztyp]['wl_1'],trcrs[nztyp]['wl_1'],ells)*lf_ll**2
        cl_ccl[nztyp]['wl_1/wl_2']=ccl.angular_cl(cosmo_cls,
                                           trcrs[nztyp]['wl_1'],trcrs[nztyp]['wl_2'],ells)*lf_ll**2
        cl_ccl[nztyp]['wl_2/wl_2']=ccl.angular_cl(cosmo_cls,
                                           trcrs[nztyp]['wl_2'],trcrs[nztyp]['wl_2'],ells)*lf_ll**2

        #Cosmic variance errors
        el_ccl[nztyp]={}
        el_ccl[nztyp]['wl_1/wl_1']=np.sqrt((cl_ccl[nztyp]['wl_1/wl_1']*cl_ccl[nztyp]['wl_1/wl_1']+
                                            cl_ccl[nztyp]['wl_1/wl_1']**2)/(2*ells+1.))
        el_ccl[nztyp]['wl_1/wl_2']=np.sqrt((cl_ccl[nztyp]['wl_1/wl_1']*cl_ccl[nztyp]['wl_2/wl_2']+
                                            cl_ccl[nztyp]['wl_1/wl_2']**2)/(2*ells+1.))
        el_ccl[nztyp]['wl_2/wl_2']=np.sqrt((cl_ccl[nztyp]['wl_2/wl_2']*cl_ccl[nztyp]['wl_2/wl_2']+
                                            cl_ccl[nztyp]['wl_2/wl_2']**2)/(2*ells+1.))

    #And make a figure that compares the results, plotting the fractional relative difference
    #between benchmark angular spectra and CCL output. We make each figure in a separate block below
    ltypes={'analytic':'-','histo':'--'}    
    
    return cl_ccl[nztyp]['wl_1/wl_1']
    