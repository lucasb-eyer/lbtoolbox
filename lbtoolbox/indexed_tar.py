from os.path import getsize
import sys
import tarfile


def index_tar_dict(fname, files_only=True, verbose=True, fail_on_duplicate=False):
    # NOTE: Duplicate entries are valid in tar files, they mean the former
    #       file has been updated by the latter one:
    #       https://www.gnu.org/software/tar/manual/html_node/multiple.html
    try:
        tarsize = getsize(fname)
    except TypeError:
        tarsize = None

    with tarfile.open(fname, mode='r:') as tar:  # This mode forbids compression.
        index = {}
        for i, tarinfo in enumerate(tar, 1):
            if files_only and not tarinfo.isfile():
                continue

            if fail_on_duplicate:
                assert tarinfo.name not in index

            index[tarinfo.name] = (tarinfo.offset_data, tarinfo.size)

            if verbose:
                sys.stderr.write("\r{}".format(i))
                if tarsize:
                    sys.stderr.write(" [{:.2%}]".format(tarinfo.offset_data/tarsize))
                sys.stderr.flush()

    return index


def index_tar_csv(fname, outfile, files_only=True, verbose=True):
    # NOTE: Duplicate entries are valid in tar files, they mean the former
    #       file has been updated by the latter one:
    #       https://www.gnu.org/software/tar/manual/html_node/multiple.html
    try:
        tarsize = getsize(fname)
    except TypeError:
        tarsize = None

    with tarfile.open(fname, mode='r:') as tar:  # This mode forbids compression.
        for i, tarinfo in enumerate(tar, 1):
            if files_only and not tarinfo.isfile():
                continue

            outfile.write('{},{},{}\n'.format(tarinfo.name, tarinfo.offset_data, tarinfo.size))

            if verbose:
                sys.stderr.write("\r{}".format(i))
                if tarsize:
                    sys.stderr.write(" [{:.2%}]".format(tarinfo.offset_data/tarsize))
                sys.stderr.flush()


def tar_index_to_dict(index_fname, fail_on_duplicate=False, hierarchy=False):
    index = {}

    with open(index_fname) as f:
        for line in f:
            name, offset, size = line.split(',')

            d = index
            if hierarchy:
                path, name = os.path.split(os.path.normpath(name))
                for component in path.split(os.path.sep):
                    d = d.setdefault(component, dict())

            if fail_on_duplicate:
                assert name not in d

            d[name] = (int(offset), int(size))

    return index


if __name__ == '__main__':
    from argparse import ArgumentParser, FileType

    parser = ArgumentParser(description='Create an index for an uncompressed tar file.')
    parser.add_argument('-t', '--tar', default=sys.stdin,
        help='The (uncompressed) tar file to be indexed. Use stdin if not given.')
    parser.add_argument('-o', '--out', default=sys.stdout, type=FileType('w+'),
        help='Path to the index file to be created. Use stdout if not given.')
    parser.add_argument('-f', '--files-only', action='store_true',
        help='Do not add directories to the index, only files.')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='Do not show progress information. Can speed up for many small entries.')
    args = parser.parse_args()

    index_tar_csv(args.tar, args.out, args.files_only, not args.quiet)
