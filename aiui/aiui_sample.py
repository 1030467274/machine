# -*- coding:utf-8 -*-
import data_accessor
import aiui_accessor
import json

question = input("请输入你要问题的问题，输入q退出程序：")
while question != 'q':
    result = json.loads(aiui_accessor.get_keywords(question))
    result = result["data"][0]["intent"]["text"]
    parameter = 0
    if result == '今天':
        parameter = 1
    elif result == '昨天':
        parameter = 0
    else:
        parameter = 2
    data_accessor.query("select * from movieschedule where playschedule=%s", [parameter])
    question = input("请输入你要问题的问题，输入q退出程序：")
