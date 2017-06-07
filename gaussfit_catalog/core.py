import numpy as np
import warnings
from astropy import units as u
from astropy.modeling import models, fitting
from radio_beam import Beam
import regions
from astropy import wcs
from astropy import coordinates
from astropy.io import fits
from astropy.stats import mad_std
from astropy.utils.console import ProgressBar
import pylab as pl
import os
import signal, sys, time

def signal_handler(signal, frame):
    # this isn't strictly necessary, but I found that the loop wasn't
    # respecting my ctrl-C's, and I DEMAND my ctrl-C's be heard!!
    sys.exit(0)

STDDEV_TO_FWHM = np.sqrt(8*np.log(2))

def bbox_contains_bbox(bbox1,bbox2):
    """returns true if bbox2 is inside bbox1"""
    return ((bbox1.ixmax>bbox2.ixmax) & (bbox1.ixmin<bbox2.ixmin) &
            (bbox1.iymax>bbox2.iymax) & (bbox1.iymin<bbox2.iymin))

def sub_bbox_slice(bbox1, bbox2):
    """returns a slice from within bbox1 of bbox2"""
    if not bbox_contains_bbox(bbox1, bbox2):
        raise ValueError("bbox2 is not within bbox1")
    x0, dx = bbox2.ixmin-bbox1.ixmin, bbox2.ixmax-bbox2.ixmin
    y0, dy = bbox2.iymin-bbox1.iymin, bbox2.iymax-bbox2.iymin
    return (slice(y0, y0+dy), slice(x0, x0+dx),)

def slice_bbox_from_bbox(bbox1, bbox2):

    if bbox1.ixmin < bbox2.ixmin:
        blcx = bbox2.ixmin
    else:
        blcx = bbox1.ixmin
    if bbox1.ixmax > bbox2.ixmax:
        trcx = bbox2.ixmax
    else:
        trcx = bbox1.ixmax
    if bbox1.iymin < bbox2.iymin:
        blcy = bbox2.iymin
    else:
        blcy = bbox1.iymin
    if bbox1.iymax > bbox2.iymax:
        trcy = bbox2.iymax
    else:
        trcy = bbox1.iymax

    y0_1 = max(blcy-bbox1.iymin,0)
    x0_1 = max(blcx-bbox1.ixmin,0)
    y0_2 = max(blcy-bbox2.iymin,0)
    x0_2 = max(blcx-bbox2.ixmin,0)

    dy_1 = min(bbox1.iymax-blcy,trcy-blcy)
    dx_1 = min(bbox1.ixmax-blcx,trcx-blcx)
    dy_2 = min(bbox2.iymax-blcy,trcy-blcy)
    dx_2 = min(bbox2.ixmax-blcx,trcx-blcx)

    view1 = (slice(y0_1, y0_1+dy_1),
             slice(x0_1, x0_1+dx_1),)
    view2 = (slice(y0_2, y0_2+dy_2),
             slice(x0_2, x0_2+dx_2),)
    for slc in view1+view2:
        assert slc.start >= 0
        assert slc.stop >= 0
    return view1,view2




def gaussfit_catalog(fitsfile, region_list, radius=1.0*u.arcsec,
                     max_radius_in_beams=2,
                     max_offset_in_beams=1,
                     background_estimator=np.nanmedian,
                     noise_estimator=lambda x: mad_std(x, ignore_nan=True),
                     savepath=None,
                     prefix="",
                    ):

    # need central coordinates of each object
    coords = coordinates.SkyCoord([reg.center for reg in region_list])

    fh = fits.open(fitsfile)
    data = fh[0].data.squeeze()
    header = fh[0].header
    datawcs = wcs.WCS(header).celestial
    beam = Beam.from_fits_header(header)
    pixscale = wcs.utils.proj_plane_pixel_area(datawcs)**0.5 * u.deg
    bmmaj_px = (beam.major.to(u.deg) / pixscale).decompose()

    noise = noise_estimator(data)

    fit_data = {}

    pb = ProgressBar(len(region_list))

    for ii,reg in enumerate(region_list):

        phot_reg = regions.CircleSkyRegion(center=reg.center, radius=radius)
        pixreg = phot_reg.to_pixel(datawcs)
        mask = pixreg.to_mask()
        cutout = mask.cutout(data) * mask.data
        cutout_mask = mask.data.astype('bool')

        smaller_phot_reg = regions.CircleSkyRegion(center=reg.center,
                                                   radius=beam.major/STDDEV_TO_FWHM)
        smaller_pixreg = smaller_phot_reg.to_pixel(datawcs)
        smaller_mask = smaller_pixreg.to_mask()
        smaller_cutout = smaller_mask.cutout(data) * smaller_mask.data

        # mask out (as zeros) neighboring sources within the fitting area
        nearby_matches = phot_reg.contains(coords, datawcs)
        if any(nearby_matches):
            inds = np.where(nearby_matches)[0].tolist()
            inds.remove(ii)
            for ind in inds:
                maskoutreg = regions.CircleSkyRegion(center=region_list[ind].center,
                                                     radius=beam.major)
                mpixreg = maskoutreg.to_pixel(datawcs)
                mmask = mpixreg.to_mask()

                view, mview = slice_bbox_from_bbox(mask.bbox, mmask.bbox)
                cutout_mask[view] &= ~mmask.data.astype('bool')[mview]
                cutout = cutout * cutout_mask


        background_mask = cutout_mask.copy().astype('bool')
        background_mask[sub_bbox_slice(mask.bbox, smaller_mask.bbox)] &= ~smaller_mask.data.astype('bool')
        background = background_estimator(cutout[background_mask])

        sz = cutout.shape[0]
        mx = np.nanmax(smaller_cutout)
        ampguess = mx-background

        p_init = models.Gaussian2D(amplitude=ampguess,
                                   x_mean=sz/2,
                                   y_mean=sz/2,
                                   x_stddev=bmmaj_px/STDDEV_TO_FWHM,
                                   y_stddev=bmmaj_px/STDDEV_TO_FWHM,
                                   bounds={'x_stddev':(bmmaj_px/STDDEV_TO_FWHM*0.75,
                                                       bmmaj_px*max_radius_in_beams/STDDEV_TO_FWHM),
                                           'y_stddev':(bmmaj_px/STDDEV_TO_FWHM*0.75,
                                                       bmmaj_px*max_radius_in_beams/STDDEV_TO_FWHM),
                                           'x_mean':(sz/2-max_offset_in_beams*bmmaj_px/STDDEV_TO_FWHM,
                                                     sz/2+max_offset_in_beams*bmmaj_px/STDDEV_TO_FWHM),
                                           'y_mean':(sz/2-max_offset_in_beams*bmmaj_px/STDDEV_TO_FWHM,
                                                     sz/2+max_offset_in_beams*bmmaj_px/STDDEV_TO_FWHM),
                                           'amplitude':(ampguess*0.9, ampguess*1.1)
                                          }
                                  )

        result, fit_info, chi2 = gaussfit_image(image=(cutout-background)*mask.data,
                                                gaussian=p_init,
                                                weights=1/noise**2,
                                                plot=savepath is not None,
                                               )
        sourcename = reg.meta['text'].strip('{}')
        pl.savefig(os.path.join(savepath, '{0}{1}.png'.format(prefix, sourcename)),
                   bbox_inches='tight')

        if 'param_cov' not in fit_info:
            fit_info['param_cov'] = np.zeros([6,6])
            success = False
        else:
            success = True

        cx,cy = pixreg.bounding_box.ixmin+result.x_mean, pixreg.bounding_box.iymin+result.y_mean
        clon,clat = datawcs.wcs_pix2world(cx, cy, 0)
        fit_data[sourcename] = {'amplitude': result.amplitude,
                                'center_x': float(clon)*u.deg,
                                'center_y': float(clat)*u.deg,
                                'fwhm_x': result.x_stddev * STDDEV_TO_FWHM * pixscale.to(u.arcsec),
                                'fwhm_y': result.y_stddev * STDDEV_TO_FWHM * pixscale.to(u.arcsec),
                                'pa': (result.theta*u.rad).to(u.deg),
                                'chi2': chi2,
                                'chi2/n': chi2/mask.data.sum(),
                                'e_amplitude': fit_info['param_cov'][0,0]**0.5,
                                'e_center_x': fit_info['param_cov'][1,1]**0.5*u.deg,
                                'e_center_y': fit_info['param_cov'][2,2]**0.5*u.deg,
                                'e_fwhm_x': fit_info['param_cov'][3,3]**0.5 * STDDEV_TO_FWHM * pixscale.to(u.arcsec),
                                'e_fwhm_y': fit_info['param_cov'][4,4]**0.5 * STDDEV_TO_FWHM * pixscale.to(u.arcsec),
                                'e_pa': fit_info['param_cov'][5,5]**0.5 * u.deg,
                                'success': success,
                               }

        pb.update(ii)
        signal.signal(signal.SIGINT, signal_handler)

    return fit_data


def gaussfit_image(image, gaussian, weights=None,
                   fitter=fitting.LevMarLSQFitter(), plot=False):

    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fitted = fitter(gaussian, xx, yy, image, weights=weights,
                        maxiter=1000)

    fitim = fitted(xx,yy)
    residual = image-fitim
    residualsquaredsum = np.nansum(residual**2*weights)

    if plot:
        pl.clf()
        ax1 = pl.subplot(2,2,1)
        im = ax1.imshow(image, cmap='viridis', origin='lower',
                        interpolation='nearest')
        vmin, vmax = im.get_clim()
        ax2 = pl.subplot(2,2,2)
        ax2.imshow(fitim, cmap='viridis', origin='lower',
                   interpolation='nearest', vmin=vmin, vmax=vmax)
        ax3 = pl.subplot(2,2,3)
        ax3.imshow(residual, cmap='viridis', origin='lower',
                   interpolation='nearest', vmin=vmin, vmax=vmax)
        ax4 = pl.subplot(2,2,4)
        im = ax4.imshow(image, cmap='viridis', origin='lower',
                        interpolation='nearest')
        vmin, vmax = im.get_clim()
        ax4.contour(fitim, levels=np.array([0.00269, 0.0455, 0.317])*fitim.max(),
                    colors=['w']*4)
        axlims = ax4.axis()
        ax4.plot(fitted.x_mean, fitted.y_mean, 'w+')
        ax4.axis(axlims)
    
    return fitted, fitter.fit_info, residualsquaredsum
