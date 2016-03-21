import numpy as np

def zca(dat):
    N = dat.shape[0]
    X = dat.reshape((N, -1)) / 255.0
    D = X.shape[1]
    xm = np.mean(X, axis=0)
    X -= xm

    # eigenvalue decomposition of the covariance matrix
    C = np.dot(X.T, X) / N
    U, lam, V = np.linalg.svd(C)  # U[:, i] is the i-th eigenvector
    print(U.shape)
    print(V.shape)
    print(lam.shape)
    # for i in range( D ):
    #    print i, lam[i], lam[i] / lam[0]

    # ZCA whitening
    eps = 0
    sqlam = np.sqrt(lam + eps)
    Uzca = np.dot(U / sqlam[np.newaxis, :], U.T)
    print(Uzca.shape)
    Z = np.dot(X, Uzca.T)
    return Z


def construct_image(X_Matrix, Y_Matrix, ImageName):
    w = h = 32
    nclass = 10
    nimg = 10
    gap = 4

    width = nimg * (w + gap) + gap
    height = nclass * (h + gap) + gap
    img = np.zeros((height, width, 3), dtype=int) + 128

    for iy in range(nclass):
        lty = iy * (h + gap) + gap
        idx = np.where(Y_Matrix == iy)[0]
        for ix in range(nimg):
            ltx = ix * (w + gap) + gap
            tmp = X_Matrix[idx[ix], :].reshape((3, h, w))
            # BGR <= RGB
            img[lty:lty + h, ltx:ltx + w, 0] = tmp[2, :, :]
            img[lty:lty + h, ltx:ltx + w, 1] = tmp[1, :, :]
            img[lty:lty + h, ltx:ltx + w, 2] = tmp[0, :, :]

    cv2.imwrite(ImageName, img)

def construct_ZCAimage(X_Matrix, Y_Matrix, ImageName):
    w = h = 32
    nclass = 10
    nimg = 10
    gap = 4

    width  = nimg * ( w + gap ) + gap
    height = nclass * ( h + gap ) + gap
    img = np.zeros( ( height, width, 3 ), dtype = int )

    for iy in range( nclass ):
        lty = iy * ( h + gap ) + gap
        idx = np.where( Y_Matrix == iy )[0]

        for ix in range( nimg ):
            ltx = ix * ( w + gap ) + gap
            absmax = np.max( np.abs( X_Matrix[idx[ix], :] ) )
            tmp = X_Matrix[idx[ix], :].reshape( ( 3, h, w ) ) / absmax *127 + 128
            # BGR <= RGB
            img[lty:lty+h, ltx:ltx+w, 0] = tmp[2, :, :]
            img[lty:lty+h, ltx:ltx+w, 1] = tmp[1, :, :]
            img[lty:lty+h, ltx:ltx+w, 2] = tmp[0, :, :]

    cv2.imwrite( ImageName, img )
