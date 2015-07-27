import sys as _sys
import os as _os
import email.utils as _eutils

try:  # Py3
    from urllib.request import urlopen as _urlopen
    from urllib.error import URLError as _URLError
    from socket import timeout as _timeout
except ImportError:  # Py2
    from urllib2 import urlopen as _urlopen
    from urllib2 import URLError as _URLError
    FileExistsError = OSError
    FileNotFoundError = IOError


def _httpfilename(response):
    """
    Python2/3 compatibility function.

    Returns the filename stored in the `Content-Disposition` HTTP header of
    given `response`, or `None` if that header is absent.
    """
    try:  # Py3
        return response.info().get_filename()
    except AttributeError:  # Py2
        import cgi
        _, params = cgi.parse_header(response.headers.get('Content-Disposition', ''))
        return params.get('filename', None)


def _getheader(response, header, default=None):
    """
    Python2/3 compatibility function.

    Returns a HTTP `header` from given HTTP `response`, or `default` if that
    header is absent.
    """
    try:  # Py3
        return response.getheader(header, default)
    except AttributeError:  # Py2
        return response.info().getheader(header, default)


def download(url, into='~/.cache/beacon8', saveas=None, desc=None, quiet=False):
    """
    Downloads the content of `url` into a file in the directory `into`.

    - `url`: The URL to download content from.
    - `into`: The folder to save the downloaded content to.
    - `saveas`: Optionally a different filename than that from the URL.
    - `desc`: Text used for progress-description.
    - `quiet`: Suppresses any console-output of this function if `True`.
    """

    # Make sure the target folder exists.
    into = _os.path.expanduser(into)
    try:
        _os.makedirs(into)
    except FileExistsError:
        pass

    try:
        response = _urlopen(url, timeout=5)
    except (_URLError, _timeout):
        # No internet connection is available, so just trust the file if it's already there.
        saveas = saveas or _os.path.basename(url)
        target = _os.path.join(into, saveas)

        if not _os.path.isfile(target):
            raise FileNotFoundError(target)
        if not quiet:
            print("No internet connection; using untrusted cached file at {}".format(target))
        return target

    # We do have an internet connection and were able to get to the URL.
    saveas = saveas or _httpfilename(response) or _os.path.basename(url)
    target = _os.path.join(into, saveas)

    leng = int(_getheader(response, 'Content-Length', 0))
    # assert leng == response.length, "Huh, looks like we didn't get all data. Maybe retry?"

    # In case the file's already there, we may avoid re-downloading it.
    if _os.path.isfile(target):
        # First, check if we got ETag which is a widely-supported checksum-ish HTTP header.
        try:
            with open(target + '.etag', 'r') as f:
                etag = f.read()
            if _getheader(response, 'ETag') == etag:
                return target
        except FileNotFoundError:
            pass

        # Alternatively, check whether the file has the same size.
        if _os.path.getsize(target) == leng:
            # If there's no last-modified header, just trust it blindly.
            servertime = _eutils.parsedate_tz(_getheader(response, 'Last-Modified'))
            if servertime is None:
                return target
            else:
                # But if there is, we may also check that.
                if _os.path.getmtime(target) >= _eutils.mktime_tz(servertime):
                    return target

    # TODO: Use progressbar from example utils.
    if not quiet:
        desc = desc or '{} to {}'.format(url, target)
        _sys.stdout.write('Downloading {}: {}k/{}k (:.2%)'.format(desc, 0, leng//1024, 0))
        _sys.stdout.flush()

    with open(target, 'wb+') as f:
        while f.tell() < leng:
            f.write(response.read(1024*8))
            if not quiet:
                _sys.stdout.write('\rDownloading {}: {}k/{}k ({:.2%})'.format(desc, f.tell()//1024, leng//1024, float(f.tell())/leng))
                _sys.stdout.flush()
    if not quiet:
        print("")

    # Finally, if present, save the ETag for later checking.
    etag = _getheader(response, 'ETag')
    if etag is not None:
        with open(target + '.etag', 'w+') as f:
            f.write(etag)

    return target
