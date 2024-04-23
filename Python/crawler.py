# user_input_crawler.py
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class MySpider(CrawlSpider):
    name = 'my_spider'

    def __init__(self, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.seed_urls = ['https://en.wikipedia.org/wiki/Main_Page']
        self.max_pages = int(input("Enter maximum number of pages to crawl (or enter 0 for unlimited): "))
        self.max_depth = int(input("Enter maximum depth of crawling (or enter 0 for unlimited): "))
        self.visited_pages = set()

    def start_requests(self):
        for url in self.seed_urls:
            yield scrapy.Request(url.strip(), callback=self.parse, meta={'depth': 1})

    def parse(self, response):
        self.visited_pages.add(response.url)
        title = response.css('title::text').get()
        content = response.css('p::text').getall()
        yield {
            'url': response.url,
            'title': title,
            'content': content
        }

        if self.max_pages and len(self.visited_pages) >= self.max_pages:
            self.logger.info(f"Reached maximum pages limit: {self.max_pages}. Crawling stopped.")
            return

        if self.max_depth and response.meta['depth'] >= self.max_depth:
            self.logger.info(f"Reached maximum depth limit: {self.max_depth}. Crawling stopped.")
            return

        links = LinkExtractor(allow=()).extract_links(response)
        for link in links:
            if link.url not in self.visited_pages:
                yield response.follow(link, callback=self.parse, meta={'depth': response.meta['depth'] + 1})

    rules = (
        Rule(LinkExtractor(allow=()), callback='parse', follow=True),
    )
