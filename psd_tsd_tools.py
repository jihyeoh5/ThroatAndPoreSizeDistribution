import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def parse_histogram(h, voxel_size):
    r"""
    Calculates the bins of the histogram that can be used to plot the 
    probability density function ('pdf') or cumulative distribution function
    ('cdf'). 
    
    Parameters
    ----------
    h : ND-array
        Contains values and bin edges of the histogram.
    voxel_size : int
        Voxel length of the image.
    
    Returns
    -------
    hist : class
        Contains the pdf, cdf, bin_centers, bin_widths of the histogram.
    """
    
    delta_x = h[1]
    P = h[0]
    temp = P * (delta_x[1:] - delta_x[:-1])
    
    class Results: ...
    hist = Results()
    hist.pdf = P
    hist.cdf = np.cumsum(temp[-1::-1])[-1::-1]
    hist.relfreq = P * (delta_x[1:] - delta_x[:-1])
    hist.bin_centers = ((delta_x[1:] + delta_x[:-1]) / 2) * voxel_size
    hist.bin_edges = delta_x * voxel_size
    hist.bin_widths = (delta_x[1:] - delta_x[:-1]) * voxel_size
    return hist

def dist_curvefit(dist,sd,voxel_size): 
    r"""   
    Calculates shape, location, and scale parameter of both the throat
    or pore size distribution ('sd') given the specified continuous 
    distribution and voxel size.
    
    This function fits a continuous distribution to the cumulative
    distribution function (cdf) of the tsd/psd. It returns the parameters
    of the continuous distribution and the corresponding fitted curve, which
    is used for plotting.
    
    Parameters
    ----------
    dist : scipy.states._continuous_distns
        Function of a continuous distribution, ex. weibull_min, lognorm, gamma.
    sd : class
        Represents the throat or size distribution.
    voxel_size : int
        Voxel length of the image.
        
    Returns
    -------
    pt : class
        Represents the pore or throat size distribution, holding three class attributes: 
        cdf, parameters of the fitted continuous distribution, and the 
        corresponding fitted curve. 
    """

    bars = 1-np.array(sd.cdf) 
    binsfit = np.array(sd.bin_centers)/voxel_size
   
    def func(bins,c0,n0,loc0): 
        try:
            return dist.cdf(bins,c0,scale=n0,loc=loc0)
        except:
            return dist.cdf(bins,scale=n0,loc=loc0)
        
    param,pcov = curve_fit(func,binsfit,bars)
    
    if len(param)==3:
        fit = func(binsfit,param[0],param[1],param[2])
    else: 
        fit = func(binsfit,param[0],param[1])
        
    class Results: ...
    pt = Results()
    pt.cdf = bars
    pt.param = param
    pt.fit = fit
    return pt

def generate_pn(network,geo,dist,pparam,voxel_size):
    r"""   
    This function produces the pore size distribution for the porous network
    using the provided shape, location, and scale parameter.
    
    Random pore seeds are generated, then percentile point function (ppf) of 
    the specified continuous distribution determines the pore sizes. 
    
    Parameters
    ----------
    network : OpenPNM.network
        Represents the cubic porous network.
    geo : OpenPNM.geometry
        Represents the geometry of the porous network.
    dist : scipy.states._continuous_distns
        Function of a continuous distribution, ex. weibull_min, lognorm, gamma.
    pparam : ND-array
        The shape, scale, and location parameters, or in the case of normal 
        continuous distribution, scale and location parameters only.
    voxel_size : int
        Voxel length of the image.
        
    Returns
    -------
    pn : class
        Represents the pore size distribution of the porous network, holding 
        class attributes: pdf, cdf, bin_centers, and binwidth. 
    geo : OpenPNM.geometry
        Represents the geometry of the porous network, with newly defined pore
        size distribution. 
    """
    
    geo['pore.seed'] = np.random.rand(network.Np)
    
    if len(pparam)==3:
        r = dist.ppf(geo['pore.seed'], c=pparam[0], scale=pparam[1], loc=pparam[2])  
    else:
        r = dist.ppf(geo['pore.seed'],scale=pparam[0], loc=pparam[1])
        
    geo['pore.diameter'] = r*2*voxel_size 
    plt.hist(geo['pore.diameter'], density=True)
    H = plt.hist(geo['pore.diameter'],density=True)
    H = parse_histogram(H,voxel_size=1)
   
    class Results: ...
    pn = Results()
    pn.binwidth = H.bin_widths[0]
    pn.bins = np.array(H.bin_centers)
    pn.pdf = H.pdf
    pn.cdf = 1-np.array(H.cdf) 
    return pn,geo

def generate_tn(network,geo,dist,tparam,voxel_size):
    r"""   
    This function produces the throat size distribution for the porous network
    using the provided shape, location, and scale parameter. 
    
    Random throat seeds are generated, then percentile point function (ppf) of 
    the specified continuous distribution determines the throat sizes.
    
    Parameters
    ----------
    network : OpenPNM.network
        Represents the cubic porous network.
    geo : OpenPNM.geometry
        Represents the geometry of the porous network.
    dist : scipy.states._continuous_distns
        Function of a continuous distribution, ex. weibull_min, lognorm, gamma.
    pparam : ND-array
        The shape, scale, and location parameters, or in the case of normal 
        continuous distribution, scale and location parameters only.
    voxel_size : int
        Voxel length of the image.
        
    Returns
    -------
    tn : class
        Represents the throat size distribution of the porous network, holding 
        class attributes: pdf, cdf, bin_centers, and binwidth. 
    geo : OpenPNM.geometry
        Represents the geometry of the porous network, with newly defined throat
        size distribution. 
    """
    
    geo['throat.seed'] = np.random.rand(network.Nt)
    
    if len(tparam)==3:
        r = dist.ppf(q=geo['throat.seed'], c=tparam[0], scale=tparam[1], loc=tparam[2]) 
    else:
        r = dist.ppf(q=geo['throat.seed'],scale=tparam[0], loc=tparam[1])
        
    geo['throat.diameter'] = 2*voxel_size*r
    H = plt.hist(geo['throat.diameter'],density=True)
    H = parse_histogram(H,voxel_size=1)
    
    class Results:...
    tn = Results()
    tn.binwidth = H.bin_widths[0]
    tn.bins = np.array(H.bin_centers)
    tn.pdf = H.pdf
    tn.cdf = 1-np.array(H.cdf)
    return tn, geo

def switch_throat(geo):
    r"""
    Manipulates the throat size distribution to ensure that all throat sizes
    are smaller than its neighboring pores. Throat sizes are mostly swapped, opposed to
    introducing new throat sizes, so that the throat size distribution can be 
    preserved. 
    
    The nearest smaller throat size is identified and are swapped in placement. In
    the minority case where there are no available smaller throat sizes, they are 
    iteratively shrunk in size (by 20%) until it is adequately small. 
    
    Parameters
    ----------
    geo : OpenPNM.geometry
        Represents the geometry of the porous network, containing the throat size
        distribution.
    Returns
    -------
    geo : OpenPNM.geometry
        Represents the geometry of the porous network, with revised throat diameters.
    """
    
    mask = np.where(geo['throat.diameter']>geo['throat.maxsize'])[0]

    for i in range(0,len(mask)):
        swap = geo['throat.diameter'][mask[i]]
        maxsize = geo['throat.maxsize'][mask[i]]
        tsizes_dict = {geo['throat.diameter'][mask[j]]:mask[j] for j in range(i+1, len(mask))}
        tsizes = list(tsizes_dict.keys())
        tsizes.sort()
        smallerthroats = [x for x in tsizes if maxsize>x]

        if smallerthroats == []:
            while geo['throat.diameter'][mask[i]] > geo['throat.maxsize'][mask[i]]:
                geo['throat.diameter'][mask[i]] = swap/1.2
                swap = swap/1.2
        else:
            newthroat = smallerthroats[-1]
            geo['throat.diameter'][mask[i]] = newthroat
            geo['throat.diameter'][tsizes_dict[newthroat]] = swap
    return geo
