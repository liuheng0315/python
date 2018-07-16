#!/usr/bin/env python
# -*- coding: utf-8 -*-
from urllib import request
import re
from xlwt import *
class IPanelSpider():
    url='http://www.ipanel.cn/article.php?code=00090001'
    htmlPattern='<div class="list">([\s\S]*?)</div>'
    datePattern = '<span class="date">([\s\S]*?)</span>([\s\S]*?)'
    articlePattern='<li><a[\s\S]*?>([\s\S]*?)</a><img'
    titleDatePattern='<li class="list_title">([\s\S]*?)</li>'
    titlePattern='([\s\S]*?)<span class="date">'
    newsdatePattern='/><span class="date">([\s\S]*?)</span>'
    #抓取页面内容
    def __fetch_content(self):
        r=request.urlopen(IPanelSpider.url)
        htmls=r.read()
        htmls=str(htmls,encoding='utf-8')
        return htmls
    #解析页面标题
    def __anysisTitle(self,htmls):
        ipanelHtml = re.findall(IPanelSpider.htmlPattern, htmls)
        ipanelHtml = ''.join(ipanelHtml)
        titleDateList=[]
        titleDateArray=re.findall(IPanelSpider.titleDatePattern,ipanelHtml)
        #正则获取，输出为dict格式
        for titleDate in titleDateArray:
            title=re.findall(IPanelSpider.titlePattern,titleDate)
            date=re.findall(IPanelSpider.datePattern,titleDate)
            titleDatedict={'title':title,'date':date}
            titleDateList.append(titleDatedict)
        return titleDateList
    #解析新闻列表
    def __ansisyNews(self,htmls):
        ipanelHtml = re.findall(IPanelSpider.htmlPattern, htmls)
        newsArticle=[]
        # 正则获取，输出为dict格式
        for html in ipanelHtml:
            article=re.findall(IPanelSpider.articlePattern,html)
            date = re.findall(IPanelSpider.newsdatePattern, html)
            news={'article':article,'date':date}
            newsArticle.append(news)
        return newsArticle
    #写入Excel
    def __writeExcel(self,titleDateList,newsArticle):
        file=Workbook(encoding='utf-8')
        table=file.add_sheet('iPanelData')
        #行数i
        i=0
        j=0
        for titleDate in titleDateList:
            title=titleDate['title'][0]
            date=titleDate['date'][0]
            table.write(i, j, title)
            j = j+1
            table.write(i, j, date)
        for news in newsArticle:
            i=0
            for article in news['article']:
                i = i + 1
                j = 0
                table.write(i, j, article)
            k=0
            for date in news['date']:
                k=k+1
                j=1
                table.write(k, j, date)
        file.save('iPanelData.xlsx')

    def go(self):
        htmls = self.__fetch_content()
        titleDateList = self.__anysisTitle(htmls)
        newsArticle = self.__ansisyNews(htmls)
        self.__writeExcel(titleDateList, newsArticle)
spider=IPanelSpider()
spider.go()