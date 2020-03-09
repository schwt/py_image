#!/usr/bin/env python
# -*- coding: utf8 -*-
import arrow

class Loger():
    def __init__(self):
        self.ts = arrow.now().timestamp
        # self.form = 'YYYY-MM-DD HH:mm:ss'
        self.form = 'HH:mm:ss'

    def log(self, info):
        self.ts = arrow.now().timestamp
        str_now  = arrow.now().format(self.form)
        print("[%s] %s" % (str_now, info))

    def logu(self, info):
        ts_diff = arrow.now().timestamp - self.ts 
        self.ts = arrow.now().timestamp
        str_now  = arrow.now().format(self.form)
        str_used = arrow.get(ts_diff).format(self.form)
        if ts_diff < 61:
            str_used = str_used[-2:]
        elif ts_diff < 3601:
            str_used = str_used[-5:]
        else:
            str_used = str_used[-8:]
        print("[%s] (%s) %s" % (str_now, str_used, info))

