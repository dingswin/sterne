####################################################################
## Codes written for general use
## Hao Ding
####################################################################

import math,os,sys
import numpy as np

class Convert_time:
    def __init__(self):
        pass
    def decyear2mjd(self, decyear):
        from astropy.time import Time
        t = Time(decyear, format='decimalyear')
        return t.mjd

def simulate_split_normal_distribution(mu, err_up, err_low, size):
    pass

def upper_limit_or_lower_limit_with_larger_magnitude(value, err):
    if value < 0:
        output = value - abs(err)
    else:
        output = value + abs(err)
    return output
def upper_limit_or_lower_limit_with_smaller_magnitude(value, err):
    if value < 0:
        output = value + abs(err)
    else:
        output = value - abs(err)
    return output

def table_str(string_in_table):
    a = str(string_in_table)
    b = a.split('\n')[-1].strip()
    return b

def dms_str2deg(string): #-- convert dd:mm:ss.ssss to dd.dddd
    if ':' in string:
        a = string.split(':')
    else:
        a = string.split(' ')
    sign1 = 1
    if a[0].strip().startswith('-'):
        sign1 = -1
    b = []
    for item in a:
        b.append(float(item))
    if b[0] < 0:
        sign = -1
    elif b[0] == 0 and sign1 == -1:
        sign = -1
    else:
        sign = 1
    b[0] = abs(b[0])
    i=len(b)
    degree = b[i-1]
    while i != 1:
        degree /= 60
        i -= 1
        degree += b[i-1]
    degree *= sign
    #degree = ((b[2]/60+b[1])/60+b[0])*sign
    return degree
def dms2deg(array):
    if type(array)==str:
        degrees = dms_str2deg(array)
    else:
        degrees = np.array([])
        for string in array:
            degree = dms_str2deg(string)
            degrees = np.append(degrees, degree)
    return degrees

def shift_position(RA, Dec, RA_shift, Dec_shift):
    """
    Functionality
    -------------
    shift position RA by RA_shift, and Dec by Dec_shift.

    Input parameters
    ----------------
    RA : str
        in HH:MM:SS.SSSS
    Dec : str
        in dd:mm:ss.sss
    RA_shift : float
        in mas.
    Dec_shift : float
        in mas.
    """
    RA_h, Dec_deg = dms2deg(RA), dms2deg(Dec)
    Dec_rad = Dec_deg * np.pi/180.
    RA_shift_ms = RA_shift/15./np.cos(Dec_rad)
    RA_shift_h = RA_shift_ms/1000./3600.
    Dec_shift_deg = Dec_shift/1000./3600.
    RA_shifted = deg2dms(RA_h + RA_shift_h)
    Dec_shifted = deg2dms(Dec_deg + Dec_shift_deg)
    return RA_shifted, Dec_shifted

def separation_large_scale(RA1,Dec1,RA2,Dec2): 
    """
    Function
    --------
    calculate angular separation (in arcmin) on the unit spherical surface, given RAs/Decs in dd:mm:ss.ssss format.

    Caveat
    ------
    the mathematical formalism reflects the geometry on the spherical surface; it is more accurate than separation().
    However, owing to the variable precision limit from python, at small scales, separation_large_scale() is vulnerable to 
    problems. For example, cos(1e-9) becomes straight 1.
    """
    RA0 = np.array([dms2deg(str(RA1)), dms2deg(str(RA2))])
    Dec = np.array([dms2deg(str(Dec1)), dms2deg(str(Dec2))])
    Dec_rad = Dec*math.pi/180
    RA = RA0*15 #hour to deg
    RA_rad = RA*math.pi/180
    cos_sep = np.cos(Dec_rad[0])*np.cos(Dec_rad[1])*np.cos(RA_rad[0]-RA_rad[1]) + np.sin(Dec_rad[0])*np.sin(Dec_rad[1])
    sep = math.acos(cos_sep)
    sep = sep*180/math.pi*60 # rad to arcmin
    return sep
def separation(RA1,Dec1,RA2,Dec2): 
    """
    Function
    --------
    calculate angular separation (in arcmin) given RAs/Decs in dd:mm:ss.ssss format;
    
    Mathematical Formalism
    ----------------------
    When the RA offset and Dec offset are both >6 arcmin, the separation_large_scale() is used.
    Otherwise, the mathematical formalism only reflects Euclidean geometry, 
    instead of the one on the spherical surface.
    """
    RA0 = np.array([dms2deg(str(RA1)), dms2deg(str(RA2))])
    Dec = np.array([dms2deg(str(Dec1)), dms2deg(str(Dec2))])
    Dec_rad = Dec*math.pi/180.
    RA = RA0*15 #hour to deg
    if (abs(RA[1]-RA[0]) > 0.1) and (abs(Dec[1]-Dec[0]) > 0.1): ## large scale
        RA_rad = RA*math.pi/180
        cos_sep = np.cos(Dec_rad[0])*np.cos(Dec_rad[1])*np.cos(RA_rad[0]-RA_rad[1]) + np.sin(Dec_rad[0])*np.sin(Dec_rad[1])
        sep = math.acos(cos_sep)
        sep = sep*180/math.pi*60 # rad to arcmin
    else: ## small scale --> use approximation
        sep_sq = (Dec[1]-Dec[0])**2 + (RA[1]-RA[0])**2 * np.cos(Dec_rad[0]) * np.cos(Dec_rad[1])
        sep = 60*np.sqrt(sep_sq) #arcmin
    return sep
def separations(RA, Dec, RAs, Decs):
    """
    like the 'separation' function, but working for array as well.
    """
    if type(RAs) == str:
        sep = separation(str(RA), str(Dec), RAs, Decs)
        return sep
    else:
        seps = np.array([])
        for i in range(len(RAs)):
            sep = separation(str(RA), str(Dec), str(RAs[i]), str(Decs[i]))
            seps = np.append(seps, sep)
        return seps ## in arcmin

def separations_deg1(RA, Dec, RAs, Decs):
    """
    like the 'separation_deg' function, but working for array as well.
    """
    if type(RAs) == float:
        sep = separation_deg(RA, Dec, RAs, Decs)
        return sep
    else:
        seps = np.array([])
        length = len(RAs)
        count = 0
        for i in range(length):
            sep = separation_deg(RA, Dec, RAs[i], Decs[i])
            seps = np.append(seps, sep)
            print(("\x1B[1A\x1B[Kprogress:{0}%".format(round((count + 1) * 100 / length)) + " \r"))
            count += 1
        return seps
def separation_deg(RA1, Dec1, RA2, Dec2): # all in deg format
    Dec = np.array([Dec1, Dec2])
    RA = np.array([RA1, RA2])
    Dec_rad = Dec*math.pi/180
    RA_rad = RA*math.pi/180
    cos_sep = math.cos(Dec_rad[0])*math.cos(Dec_rad[1])*math.cos(RA_rad[0]-RA_rad[1]) + math.sin(Dec_rad[0])*math.sin(Dec_rad[1])
    sep = math.acos(cos_sep)
    sep = sep*180/math.pi # rad to deg
    return sep #in deg
def separations_deg(RA, Dec, RAs, Decs):
    """
    like the 'separation_deg' function, but working for array as well.

    Input parameters
    ----------------
    all in deg

    Output parameters
    -----------------
    seps : float/list of float
        angular separation(s) (in deg).
    """
    if type(RAs) == float:
        sep = separation_deg(RA, Dec, RAs, Decs)
        return sep
    else:
        RA_rad, Dec_rad, RAs_rad, Decs_rad = RA*math.pi/180, Dec*math.pi/180, RAs*math.pi/180, Decs*math.pi/180
        cos_seps = np.cos(Dec_rad)*np.cos(Decs_rad)*np.cos(RA_rad-RAs_rad) + np.sin(Dec_rad)*np.sin(Decs_rad)
        seps = np.arccos(cos_seps)
        seps *= 180/math.pi
        return seps ## in deg



def colonizedms(string):
    import re
    string = string.strip()
    sign = ''
    if string.startswith('-') or string.startswith('+'):
        sign = string[0]
        string = string[1:]
    #    sign = -1
    #else:
    #    sign = 1
    #string = filter(str.isdigit,string)
    newstr = re.sub('\s+', ':', string.strip())
    #newstr = string[:2] + ':' + string[2:4] + ':' + string[4:6] + '.' + string[6:]
    #if sign == -1:
    #    newstr = "-" + newstr
    newstr = sign + newstr
    return newstr

def deg2dms(array):
    if type(array)==float or type(array)==np.float64:
        dmss = deg_float2dms(array)
    else:
        dmss = np.array([])
        for number in array:
            dms = deg_float2dms(number)
            dmss = np.append(dmss, dms)
        if len(dmss) == 1:
            dmss = dmss[0]
    return dmss
def deg_float2dms(degree): #-- degree to dd:mm:ss.sssssss
    sign = np.sign(degree)
    degree = float(abs(degree)) 
    d = math.floor(degree)
    m = math.floor((degree-d)*60)
    s = ((degree-d)*60-m)*60
    dms="%02d:%02d:%010.7f" % (d,m,s)
    if sign == -1:
        dms = '-' + dms
    return dms 


def mas2ms(ErrArray_mas,Dec):
    d = dms2deg(Dec)*math.pi/180
    ErrArray_ms = ErrArray_mas/15/math.cos(d)
    return ErrArray_ms
def ms2mas(ErrArray_ms, Dec):
    d = dms2deg(Dec)*math.pi/180
    d = np.array(d)
    ErrArray_ms = np.array(ErrArray_ms)
    ErrArray_mas = ErrArray_ms * 15 * np.cos(d)
    return ErrArray_mas

def sample2estimate(array1,confidencelevel):
    array = sorted(array1)
    CL = confidencelevel
    if CL<1:
        SV = int(CL*len(array)) #SV -> significant volume
    elif CL>=1 and CL<10:
        CL = math.erf(CL/2**0.5)
        SV = int(CL*len(array))
    elif CL>=10:
        SV = CL    
    delta = float('inf')
    for i in range(len(array)-SV-1):
        diff = array[i+SV] - array[i]
        if diff < delta:
            j=i
            delta = diff
    confidence_min = array[j]
    confidence_max = array[j+SV]
    value = 0.5*(confidence_min+confidence_max)
    error = 0.5*(confidence_max-confidence_min)
    return value,error

def sample2estimate_and_median(array1,confidencelevel):
    """
    find the narrowest confidence interval and report the median of the this interval
    """
    array = sorted(array1)
    CL = confidencelevel
    if CL<1:
        SV = int(CL*len(array)) #SV -> significant volume
    elif CL>=1 and CL<10:
        CL = math.erf(CL/2**0.5)
        SV = int(CL*len(array))
    elif CL>=10:
        SV = CL    
    delta = float('inf')
    for i in range(len(array)-SV-1):
        diff = array[i+SV] - array[i]
        if diff < delta:
            j=i
            delta = diff
    confidence_min = array[j]
    confidence_max = array[j+SV]
    value = 0.5*(confidence_min+confidence_max)
    error = 0.5*(confidence_max-confidence_min)
    return value,error,array[int(j+SV/2.)]

def periodic_sample2estimate(alist, period=360, confidencelevel=1):
    """
    periodic sample cannot use median as the estimate.
    instead, it should adopt the most compact symmetric confidence interval
    as the estimate and the associated uncertainty.

    Output parameters
    -----------------
    median : flaot
    upper-side error of median : float
    lower-side error of median : float
    """
    alist = np.sort(np.array(alist))
    alist = alist % period
    CL = confidencelevel
    error = float('inf')
    median = None
    value = None
    for threshold in np.arange(0,period,period/360.):
        reordered_list = move_elements_larger_than_a_threshold_to_the_head_of_a_list(alist, threshold)
        value1, error1, median1 = sample2estimate_and_median(reordered_list, CL)
        if error1 < error:
            error = error1
            value = value1
            median = median1
    return median % period, value+error-median, median-(value-error)

def move_elements_larger_than_a_threshold_to_the_head_of_a_list(a_sorted_list, threshold, period=360):
    """
    a_sorted_list should be a numpy array
    """
    aSL = a_sorted_list
    length = len(aSL)
    index_threshold = (np.abs(aSL - threshold)).argmin()
    new_head_indice = np.arange(index_threshold, length)
    list_new_head = aSL[new_head_indice] - period
    list_new = np.concatenate((list_new_head, aSL[np.arange(index_threshold)]))
    return list_new

def sample2most_probable_value(array1, bins=1000):
    array = sorted(array1)
    bins = int(bins)
    [counts, values] = np.histogram(array, bins)
    index_max_count = np.argmax(counts)
    most_probable_value = 0.5*(values[index_max_count] + values[index_max_count+1])
    return most_probable_value

def sample2median(array1):
    array = sorted(array1)
    length = len(array)
    if length % 2 == 0:
        median = 0.5*(array[length//2-1] + array[length//2])
    else:
        median = array[(length-1)//2]
    return median
def sample2median_range(array1, confidencelevel):
    array = sorted(array1)
    CL = confidencelevel
    if CL<1:
        SV = int(CL*len(array)) #SV -> significant volume
    elif CL>=1 and CL<10:
        CL = math.erf(CL/2**0.5)
        SV = int(CL*len(array))
    elif CL>=10:
        SV = CL
    index_start = int((len(array)-SV)/2-1)
    index_end = index_start + SV
    return array[index_start], array[index_end]

    



def sample2uncertainty(array1,estimate,confidencelevel): #offered with an estimate and present symmetric format
    array = sorted(array1)
    CL = confidencelevel
    SV = int(CL*len(array))
    delta = float('inf')
    for i in range(len(array)-SV-1):
        diff = array[i+SV] + array[i] - 2*estimate
        diff = abs(diff)
        if diff < delta:
            j=i
            delta = diff
    uncertainty = estimate - array[j]
    return uncertainty

            


def weightX(Xs,errs_X): #both numpy array
    errs_X = 1./errs_X**2
    sum_err = sum(errs_X)
    Xbar = sum(errs_X/sum_err*Xs)
    err = 1./sum_err**0.5
    return Xbar, err

def weighted_avg_and_std(Xs, errs_X=1):
    N = len(Xs)
    if type(errs_X) != int:
        errs_X = 1./errs_X**2
        sum_err = sum(errs_X)
        w = errs_X/sum_err
    else:
        w = np.ones(N)
    Xbar = np.average(Xs, weights=w)
    variance = np.average((Xs-Xbar)**2, weights=w)
    variance *= N*1./(N-1) #if 1. is not added, then it becomes 1
    return (Xbar, math.sqrt(variance))

def calculate_median_and_its_error(Xs, errs_X=1):
    Xbar, std = weighted_avg_and_std(Xs, errs_X)
    median = np.median(Xs)
    error_median = 1.2533 * std
    return median, error_median

def is_pure_number_or_space(str):
    is_pure_number_or_space = True
    for letter in str:
        if not letter.isdigit() and letter != ' ':
            is_pure_number_or_space = False
            break
    return is_pure_number_or_space

def no_alphabet(str):
    no_alphabet = True
    for letter in str:
        if letter.isalpha():
            no_alphabet = False
            break
    return no_alphabet

