# -*- coding: utf-8 -*-
import requests
import time
import hashlib
import base64

URL = "http://openapi.xfyun.cn/v2/aiui"
APPID = "5bf92a08"
API_KEY = "7efb49dc8cb484d59c8c0fa946db49c1"
AUE = "raw"
AUTH_ID = "2049a1b2fdedae553bd03ce6f4820ac4"
DATA_TYPE = "text"
SAMPLE_RATE = "16000"
SCENE = "main"
RESULT_LEVEL = "complete"
LAT = "39.938838"
LNG = "116.368624"
# 个性化参数，需转义
PERS_PARAM = "{\\\"auth_id\\\":\\\"2049a1b2fdedae553bd03ce6f4820ac4\\\"}"
FILE_PATH = ""


#

def buildHeader():
    curTime = str(int(time.time()))
    param = "{\"result_level\":\"" + RESULT_LEVEL + "\",\"auth_id\":\"" + AUTH_ID + "\",\"data_type\":\"" + DATA_TYPE + "\",\"scene\":\"" + SCENE + "\",\"lat\":\"" + LAT + "\",\"lng\":\"" + LNG + "\"}"
    # 使用个性化参数时参数格式如下：
    # param = "{\"result_level\":\""+RESULT_LEVEL+"\",\"auth_id\":\""+AUTH_ID+"\",\"data_type\":\""+DATA_TYPE+"\",\"sample_rate\":\""+SAMPLE_RATE+"\",\"scene\":\""+SCENE+"\",\"lat\":\""+LAT+"\",\"lng\":\""+LNG+"\",\"pers_param\":\""+PERS_PARAM+"\"}"
    paramBase64 = base64.b64encode(param.encode())

    m2 = hashlib.md5()
    m2.update((API_KEY + curTime + paramBase64.decode()).encode())
    checkSum = m2.hexdigest()

    header = {
        'X-CurTime': curTime,
        'X-Param': paramBase64,
        'X-Appid': APPID,
        'X-CheckSum': checkSum,
    }
    return header


def readFile(filePath):
    # binfile = open(filePath, 'rb')
    # data = binfile.read()
    # return data
    return "今天有什么电影".encode("utf-8")


def get_keywords(txt):
    r = requests.post(URL, headers=buildHeader(), data=txt.encode("utf-8"))
    return r.content.decode()
