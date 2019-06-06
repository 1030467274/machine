# -*- coding:utf-8 -*-

import pymysql
import pymysql.cursors

HOST = 'localhost'
PORT = 3306
DATABASE = 'movies'
USER = 'root'
PASSWORD = 'Passw0rd'
CHARSET = 'utf8'


def create_connection():
    connection = pymysql.Connect(host=HOST,
                                 port=PORT,
                                 db=DATABASE,
                                 user=USER,
                                 passwd=PASSWORD,
                                 charset=CHARSET)
    return connection


def query(sql, parameters=None):
    try:
        connection = create_connection()
        cursor = connection.cursor()
        if parameters:
            cursor.execute(sql, parameters)
        else:
            cursor.execute(sql)
        result = cursor.fetchall()
        for r in result:
            print(r)
    except Exception as exception:
        print(exception)
    finally:
        cursor.close()
        connection.close()
