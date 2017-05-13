"""! This modules provides functions to compute a 2d morlet scattering network, with numpy functions only """
import numpy as np
import matplotlib.pyplot as plt

def morlet2d(j, theta, N, ksi, sigma2):
    """! Computes a Morlet wavelet in [-1, 1]x[-1, 1]
    @param j integer, the scale of the wavelet
    @param theta float, the orientation of the wavelet
    @param N integer, the size of the grid on which to sample the wavelet
    @param ksi float, the frequency of the wavelet
    @param sigma2 float, the standard deviation of the Gaussian enveloppe.
    @return a 2d numpy array, containing the wavelet
    """
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    R = np.array([[costheta, -sintheta], [sintheta, costheta]])
    sine = np.zeros((N, N), dtype=complex)
    env = np.zeros((N, N))
    #rang = np.arange(-N/2, N/2)/N*2.
    rang = np.linspace(-1, 1, N)
    C2 = 0.5
    for col, ux in enumerate(rang):
        for row, uy in enumerate(rang):
            sqnorm = (2**(2*j))*(ux**2 + uy**2)
            u_rot = (2**j)*np.dot(R, [ux, uy])
            dot = u_rot[0]*ksi
            #out[row, col] = (np.exp(1j*dot) - C2)*np.exp(-sqnorm/(2*sigma2))
            sine[row, col] = np.exp(1j*dot)
            env[row, col] = np.exp(-sqnorm/(2*sigma2))
    C2 = np.sum(sine*env)/np.sum(env)
    out = (sine - C2)*env
    out = out/np.sum(np.abs(out)**2)
    return out

def gaussian2d(N, j, sigma2):
    """! Computes the Gaussian low-pass filter on [-1, 1]x[-1, 1]
    @param N integer, the size of the grid on which the filter is sampled
    @param j the scale
    @param sigma2 the standard deviation of the Gaussian enveloppe
    @return a NxN 2d numpy array
    """
    rang = np.linspace(-1, 1, N)
    out = np.zeros((N, N))
    for col, ux in enumerate(rang):
        for row, uy in enumerate(rang):
            sqnorm = (2**(2*j))*(ux**2 + uy**2)
            #sqnorm = (ux**2 + uy**2)
            out[row, col] = np.exp(-sqnorm/(2*sigma2))
    return out

def morlet_bank(N, sigma2=0.85**2, n_angles=6, js=[1, 2, 3], ksi=3*np.pi/4):
    """! Computes the bank of Morlet wavelet and Gaussian averaging filters.
    @param N integer, the size of the output images
    @param sigma2, the variance of the Gaussian envelope of the mother wavelet,
    and the averaging filter
    @param n_angles integer, the number of orientations at each scale
    @param js, the scale of the wavelet
    @param ksi, the frequency of the mother wavelet
    @return a NxN array containing the averaging filter phi, and a
    (n_angles*len(js))xNxN array containing the wavelets
    """
    phi = gaussian2d(N, min(js), sigma2)
    psis = np.zeros((n_angles*len(js), N, N), dtype=complex)
    c = 0
    freqs = np.zeros(psis.shape[0])
    for j in js:
        for theta in np.arange(n_angles)*np.pi/n_angles:
            psis[c] = morlet2d(j, theta, N, ksi, sigma2)
            freqs[c] = j
            c += 1
    return phi, psis, freqs

def convfft2d(i1, i2):
    """! Perform a 2d circulat convolution via FFT. i1 and i1 must have the same shape """
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(i1)*np.fft.fft2(i2)))

def subsamp(a, factor):
    """! subsample an image a by a certain factor """
    return a[::factor, ::factor]

def scattering(image, phi, psis, order=1, sub_factor=8):
    """! Computes the scattering coefficients of an image at a certain order
    @param image a 2d numpy array, 1-channel image to scatter
    @param phi a 2d numpy array, the same shape as image. The low-pass filter
    @param psis a 3d numpy array, contains the wavelets. The first dimension is the wavelet index,
    dimension 1 and 2 must have the same shape as phi and image
    @param order integer, the order of the scattering network
    @sub_factor integer, the subsampling factor
    @return a list of 2d numpy array, containing the scaatering coefficients
    at each path
    """
    out = [image]
    for m in range(order):
        scatm = []
        for o in out:
            for psi in psis:
                scatm.append(np.abs(convfft2d(o, psi)))
                #plt.imshow(scatm[-1])
                #plt.show()
        out = scatm
    for i, o in enumerate(out):
        out[i] = np.abs(convfft2d(o, phi))
        out[i] = subsamp(out[i], sub_factor)
    return out

def scattering_fr_decr(image, phi, psis, js, order=1, sub_factor=8):
    """! Computes the scattering coefficients of an image at a certain order, but only
    along frequency decreasing paths. It has been proven than those paths contains
    most of the energy
    @param image a 2d numpy array, 1-channel image to scatter
    @param phi a 2d numpy array, the same shape as image. The low-pass filter
    @param psis a 3d numpy array, contains the wavelets. The first dimension is the wavelet index,
    @param js a list of integers, of length psis.shape[0], containing the
    corresponding scales of psis
    dimension 1 and 2 must have the same shape as phi and image
    @param order integer, the order of the scattering network
    @sub_factor integer, the subsampling factor
    @return a list of 2d numpy array, containing the scaatering coefficients
    at each path
    """
    out = [image]
    freqs = [100]
    for _ in range(order):
        scatm = []
        freqm = []
        for j, o in enumerate(out):
            for i, psi in enumerate(psis):
                if js[i] <= freqs[j]:
                    scatm.append(np.abs(convfft2d(o, psi)))
                    freqm.append(js[i])
        freqs = freqm
        out = scatm
    for i, o in enumerate(out):
        out[i] = np.abs(convfft2d(o, phi))
        out[i] = subsamp(out[i], sub_factor)
    return out


if __name__ == "__main__":
    from PIL import Image
    from display import big_image
    lena = np.array(Image.open("lena512.bmp"))
    print(lena.shape)
    plt.imshow(lena, cmap="gray")
    phi, psis, freqs = morlet_bank(512, js=[3, 4, 5])
    coefs = scattering_fr_decr(lena, phi, psis, freqs, 2, 32)
    plt.figure()
    plt.imshow(big_image(coefs), cmap="gray")
    plt.show()
    #plt.imshow(gaussian2d(20, 2, 0.85**2))
