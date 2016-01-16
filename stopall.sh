#!/bin/sh
kill -9 $(ps -ef|grep python|awk '{print $2}')
kill -9 $(ps -ef|grep svm|awk '{print $2}')