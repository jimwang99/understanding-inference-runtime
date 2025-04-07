import enum


class ErrCode(enum.IntEnum):
    ESUCC = 0
    ENOENT = 2
    ENOMEM = 12
    EINVAL = 22
    ENODATA = 61
    ETIME = 62
