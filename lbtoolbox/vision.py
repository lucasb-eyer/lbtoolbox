"""Computer-vision utilities that use OpenCV if available."""
import numpy as np


try:
    import cv2


    def resize_img(img, shape=None, interp=None, is_chw=False):
        """ Resize image, or copy if `shape` is `None` or same as `img`.

        Args:
            img: The image to be resized.
            shape: (h,w)-tuple of target size
            interp: None for inter-area, 'bicubic' for bicubic.
            is_chw: obvious.
        """
        if shape is None:
            return np.array(img)

        if interp is None:
            interp = cv2.INTER_AREA
        elif interp is 'bicubic':
            interp = cv2.INTER_CUBIC
        else:
            raise NotImplementedError("TODO: Interpolation {} in OpenCV".format(interp))

        if is_chw:
            img = np.rollaxis(img, 0, 3)  # CHW to HWC

        if img.shape[0] == shape[0] and img.shape[1] == shape[1]:
            return np.array(img)

        img = cv2.resize(img, (shape[1], shape[0]), interpolation=interp)

        if is_chw:
            img = np.rollaxis(img, 2, 0)  # HWC to CHW

        return img


    def resize_map(img, shape, interp='bicubic'):
        return resize_img(img, shape, interp)


    def imread(fname):
        f = cv2.imread(fname)
        if f is None:
            raise ValueError("Couldn't load file {}".format(fname))
        return f[:,:,::-1]


    def imwrite(fname, img):
        cv2.imwrite(fname, img[:,:,::-1])


    def convolve_edge_same(image, filt):
        # 64F is actually faster than 32?!
        return cv2.filter2D(image, cv2.CV_64F, filt, borderType=cv2.BORDER_REPLICATE)


    def convolve_edge_zeropad(image, filt):
        dx1, dx2 = filt.shape[1]//2, filt.shape[1]//2
        dy1, dy2 = filt.shape[0]//2, filt.shape[0]//2
        x = cv2.copyMakeBorder(image, dy1, dy2, dx1, dx2, cv2.BORDER_CONSTANT)
        x = cv2.filter2D(x, -1, filt)
        return x[dy1:-dy2,dx1:-dx2]


    def video_or_open(video):
        # Because can't access cv2.VideoCapture type (only function exposed)
        if type(video).__name__ == 'VideoCapture':
            return video
        else:
            return cv2.VideoCapture(video)


    def vidframes(video):
        return int(video_or_open(video).get(cv2.CAP_PROP_FRAME_COUNT))


    def itervid(video):
        video = video_or_open(video)

        while True:
            good, img = video.read()

            if not good:
                return

            yield img


    def vid2tensor(video, imgproc=lambda x: x, progress=None):
        video = video_or_open(video)

        T = vidframes(video)
        vid = None

        for t, img in enumerate(itervid(video)):
            img = imgproc(img)

            if vid is None:
                vid = np.empty((T,) + img.shape, img.dtype)

            vid[t] = img

            if progress is not None:
                progress(t, T)

        return vid


    def total_frames(basedir, ext='.MTS', subsample=1):
        T = 0
        for f in sane_listdir(basedir, ext=ext):
            T += vidframes(pjoin(basedir, f))//subsample

        return T


except ImportError:
    import scipy

    try:
        # This is what scipy's imread does lazily.
        from PIL import Image as _Image

        def imread(fname):
            # This does what CV_LOAD_IMAGE_ANYDEPTH does by default.
            return np.array(_Image.open(fname))

    except ImportError:

        def imread(fname):
            raise ImportError(
                "Neither OpenCV nor the Python Imaging Library (PIL) is "
                "installed. Please install either for loading images."
            )


    def resize_img(img, shape=None, interp='bilinear'):
        """ Resize image.

        Args:
            img: The image to be resized.
            shape: (h,w)-tuple of target size
            interp: 'bilinear' for bilinear, more see scipy's imresize.
        """
        if shape is None:
            return np.array(img)

        return scipy.misc.imresize(img, shape, interp=interp, mode='RGB')


    def resize_map(img, shape, interp='bicubic'):
        return scipy.misc.imresize(img, shape, interp=interp, mode='F')


    def imwrite(fname, img):
        scipy.misc.imsave(fname, img)


    def convolve_edge_same(image, filt):
        pad_width = int(filt.shape[1] / 2)
        pad_height = int(filt.shape[0] / 2)
        out_img = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
        out_img = signal.convolve2d(out_img, filt, mode='valid', boundary='fill', fillvalue=0)
        return out_img
