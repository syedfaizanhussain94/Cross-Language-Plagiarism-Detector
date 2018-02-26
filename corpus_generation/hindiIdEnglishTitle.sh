#!/bin/sh


mysql -uroot -proot << END
create database if not exists wiki;
use wiki;

# Uncomment the following to drop previous langlinks table and create a new one
# source hiwiki-20151002-langlinks.sql

END

# echo "select ll_from, ll_title from langlinks where ll_lang = 'en' and ll_title != ''" | mysql -uroot -proot wiki > hindiIdEnglishTitle

