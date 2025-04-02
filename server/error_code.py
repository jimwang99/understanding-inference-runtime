import enum


class ErrCode(enum.IntEnum):
    ESUCC = 0
    ENOMEM = 12
    EINVAL = 22
    ENODATA = 61
