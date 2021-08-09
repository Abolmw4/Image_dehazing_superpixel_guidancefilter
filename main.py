import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from skimage.segmentation import slic
from sklearn.preprocessing import MinMaxScaler
from cv2.ximgproc import rollingGuidanceFilter

from google.colab import files
uploaded = files.upload()

I=cv2.imread(r'/content/screenshot.jpg')

R, G, B = 0, 1, 2  # index for convenience
L = 256  # color depth

def Air_light(I,n,c):
  """
  Get the atmosphere light in the (RGB) image data.

    Parameters
    -----------
    I:      the M * N * 3 RGB image data ([0, L-1]) as numpy array
    n:      Number of segments for the SLIC algorithm
    c:      The compactness parameter trades off color-similarity and proximity

    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
  """
  segments = slic(I, n_segments=n, compactness=c)
  cv2_imshow(segments)
  M, N = segments.shape
  flatI = I.reshape(M * N, 3)
  flatdark = segments.ravel()
  searchidx = (-flatdark).argsort()[:M * N *c]  # find top M * N * c indexes
  print ('atmosphere light region:', [(i / N, i % N) for i in searchidx])
  
  # return the highest intensity for each channel
  return np.max(flatI.take(searchidx, axis=0), axis=0)

def get_dark_channel(I, w):
    """Get the dark channel prior in the (RGB) image data.

    Parameters
    -----------
    I:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size

    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = I.shape
    z=int(w/2)
    padded = np.pad(I,((z, z), (z, z), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  
    return darkch
  
def get_transmission(I, A, darkch, omega, w):
    """Get the transmission esitmate in the (RGB) image data.

    Parameters
    -----------
    I:       the M * N * 3 RGB image data ([0, L-1]) as numpy array
    A:       a 3-element array containing atmosphere light
             ([0, L-1]) for each channel
    darkch:  the dark channel prior of the image as an M * N numpy array
    omega:   bias for the estimate
    w:       window size for the estimate

    Return
    -----------
    An M * N array containing the transmission rate ([0.0, 1.0])
    """
    return 1 - omega * get_dark_channel(I / A, w)  
  
  
def hazeFree(I,A,T,filter=False):
  f=np.zeros(I.shape)
  for i in range(I.shape[0]):
    for j in range(I.shape[1]):
      for k in range(I.shape[2]):
        f[i,j,k]=((I[i,j,k]-A[k])/T[i,j])+A[k]
  J=f.ravel()
  data = J.reshape(-1,1)
  scaler = MinMaxScaler(feature_range=(0,255))
  print(scaler.fit(data))
  f_h=scaler.transform(data)
  final=f_h.reshape(f.shape).astype('uint8')
  if filter==True:
    dst = rollingGuidanceFilter(final, d=-1, sigmaColor=0, sigmaSpace=10, numOfIter=2)
    return dst
  else:
    return final  
  
  
  
  
  
  
  
  
  
  
  
  
  
