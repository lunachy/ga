# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:35:25 2019

@author: mayn
"""
import ctypes


class ObjPosition(ctypes.Structure):
    _fields_ = [("cid", ctypes.c_long), ("dx", ctypes.c_long), ("dy", ctypes.c_long), ("modelId", ctypes.c_long),
                ("rot", ctypes.c_int), ("x", ctypes.c_int), ("y", ctypes.c_int), ("z", ctypes.c_int),
                ("isDrModel", ctypes.c_int)]


class ObjIdentity(ctypes.Structure):
    _fields_ = [("cid", ctypes.c_long), ("dx", ctypes.c_long), ("dy", ctypes.c_long), ("modelId", ctypes.c_long),
                ("isDrModel", ctypes.c_int)]


class ConfigInfo(ctypes.Structure):
    _fields_ = [("searchWidth", ctypes.c_int), ("maxRoomBoard", ctypes.c_int), ("inputStrType", ctypes.c_int),
                ("outputMaskMode", ctypes.c_int), ("numSegs", ctypes.c_int),
                ("objGranularity", ctypes.c_int)]


class EmbeddingCode(ctypes.Structure):
    _fields_ = [("zid", ctypes.c_int), ("cid", ctypes.c_int), ("scid", ctypes.c_int), ("mid", ctypes.c_int),
                ]
    # ("did", ctypes.c_int),
    # ("rid", ctypes.c_int), ("gdid", ctypes.c_int), ("grid", ctypes.c_int)


class MultiThreadRet(ctypes.Structure):
    _fields_ = [("ret", ctypes.c_int), ("threadId", ctypes.c_int)]


class Zone(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("tag", ctypes.c_int), ("label", ctypes.c_int), ("designed", ctypes.c_bool)]


class InitResult(ctypes.Structure):
    _fields_ = [("detectorIdx", ctypes.c_int), ("numZones", ctypes.c_int)]


def init_embedding_code(element):
    element.zid = -3
    element.cid = -3
    element.scid = -3
    # element.did = -3
    # element.rid = -3
    # element.gdid = -3
    # element.grid = -3
    element.mid = -3


def init_ThreadRet_code(element):
    element.ret = -1
    element.threadId = -1


def init_InitResult_code(element):
    element.detectorIdx = -1
    element.numZones = -1
