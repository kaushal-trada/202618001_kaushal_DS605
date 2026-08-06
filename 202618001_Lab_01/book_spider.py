import scrapy

class BookSpider(scrapy.Spider):
    name = 'book_spider'
    start_urls = ['https://books.toscrape.com/catalogue/page-1.html']
    page_count = 1

    def parse(self, response):
        books = response.css('h3 a::attr(href)').getall()
        for book in books:
            yield response.follow(book, callback=self.parse_book)

        # Follow pagination up to at least 5 pages (to guarantee >100 books)
        next_page = response.css('li.next a::attr(href)').get()
        if next_page and self.page_count < 6:
            self.page_count += 1
            yield response.follow(next_page, callback=self.parse)

    def parse_book(self, response):
        yield {
            'title': response.css('div.product_main h1::text').get(),
            'category': response.xpath('//ul[@class="breadcrumb"]/li[3]/a/text()').get(),
            'price': response.css('p.price_color::text').get(),
            'rating': response.css('p.star-rating::attr(class)').get(),
            'availability': response.xpath('//th[text()="Availability"]/following-sibling::td/text()').get(),
            'description': response.xpath('//div[@id="product_description"]/following-sibling::p/text()').get(),
            'upc': response.xpath('//th[text()="UPC"]/following-sibling::td/text()').get(),
            'reviews': response.xpath('//th[text()="Number of reviews"]/following-sibling::td/text()').get(),
            'url': response.url
        }
